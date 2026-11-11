#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Execute an unranked onnx.Slice with bounds only known at run time.

The lit tests check the lowered IR. This run checks that a Slice the importer
left unranked gets the sliced extent rather than the input extent. A
destination sized from the input would add a zero-filled tail, which changes
the mean over the sliced rows.
"""

from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

_F32 = TensorProto.FLOAT
_I64 = TensorProto.INT64
_BOOL = TensorProto.BOOL


def _loop_prefix_mean_model(trip: int):
    """Scan row i is the mean of the first i+1 rows of ``x[8,4]``.

    ``x`` is a loop-carried value declared without a shape in the body, as in
    a TF export, so ORT cannot type the Slice and the importer leaves its
    result unranked.
    """
    x = helper.make_tensor_value_info("x", _F32, [8, 4])
    body_inputs = [
        helper.make_tensor_value_info("iter", _I64, []),
        helper.make_tensor_value_info("cond_in", _BOOL, []),
        helper.make_tensor_value_info("x_in", _F32, None),
    ]
    body_initializers = [
        numpy_helper.from_array(np.array(1, dtype=np.int64), "one"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "vec1"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "zero_vec"),
    ]
    body_nodes = [
        helper.make_node("Identity", ["cond_in"], ["cond_out"]),
        helper.make_node("Identity", ["x_in"], ["x_out"]),
        helper.make_node("Add", ["iter", "one"], ["end_scalar"]),
        helper.make_node("Reshape", ["end_scalar", "vec1"], ["end_vec"]),
        helper.make_node(
            "Slice", ["x_in", "zero_vec", "end_vec", "zero_vec"], ["prefix"]
        ),
        helper.make_node("ReduceMean", ["prefix"], ["row"], axes=[0], keepdims=0),
    ]
    body = helper.make_graph(
        body_nodes,
        "body",
        body_inputs,
        [
            helper.make_tensor_value_info("cond_out", _BOOL, []),
            helper.make_tensor_value_info("x_out", _F32, None),
            helper.make_tensor_value_info("row", _F32, None),
        ],
        initializer=body_initializers,
    )
    trip_init = numpy_helper.from_array(np.array(trip, dtype=np.int64), "trip")
    cond_init = numpy_helper.from_array(np.array(True), "cond")
    node = helper.make_node(
        "Loop", ["trip", "cond", "x"], ["x_final", "scan"], body=body
    )
    return make_model_from_nodes(
        [node],
        [x],
        [
            helper.make_tensor_value_info("x_final", _F32, [8, 4]),
            helper.make_tensor_value_info("scan", _F32, [None, 4]),
        ],
        initializers=[trip_init, cond_init],
    )


class TestSliceRuntimeBounds:
    def test_loop_body_prefix(self, model_runner):
        model = _loop_prefix_mean_model(3)
        x = np.arange(1, 33, dtype=np.float32).reshape(8, 4)
        actual, expected = model_runner.run_sample(model, [x])
        assert actual[1].shape == (3, 4)
        compare_outputs(actual, expected, atol=1e-5, rtol=1e-5)
