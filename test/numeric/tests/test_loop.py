#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

import numpy as np
import pytest
from onnx import TensorProto, helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes


def _append_from_empty_model(trip_count: int, width: int):
    body_iter = helper.make_tensor_value_info("iter", TensorProto.INT64, [])
    body_cond = helper.make_tensor_value_info("cond_in", TensorProto.BOOL, [])
    body_acc = helper.make_tensor_value_info("acc_in", TensorProto.FLOAT, [None, width])
    body_cond_out = helper.make_tensor_value_info("cond_out", TensorProto.BOOL, [])
    body_acc_out = helper.make_tensor_value_info(
        "acc_out", TensorProto.FLOAT, [None, width]
    )
    axes = helper.make_tensor("axes_value", TensorProto.INT64, [1], [0])
    body = helper.make_graph(
        [
            helper.make_node("Constant", [], ["axes"], value=axes),
            helper.make_node("Gather", ["X", "iter"], ["row"], axis=0),
            helper.make_node("Unsqueeze", ["row", "axes"], ["row_2d"]),
            helper.make_node("Concat", ["acc_in", "row_2d"], ["acc_out"], axis=0),
            helper.make_node("Identity", ["cond_in"], ["cond_out"]),
        ],
        "append_body",
        [body_iter, body_cond, body_acc],
        [body_cond_out, body_acc_out],
    )

    x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [trip_count, width])
    y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, width])
    initializers = [
        helper.make_tensor("M", TensorProto.INT64, [], [trip_count]),
        helper.make_tensor("cond", TensorProto.BOOL, [], [True]),
        helper.make_tensor("seed", TensorProto.FLOAT, [0, width], []),
    ]
    loop = helper.make_node("Loop", ["M", "cond", "seed"], ["Y"], body=body)
    return make_model_from_nodes(
        [loop], [x_info], [y_info], initializers=initializers, opset=17
    )


@pytest.mark.parametrize("trip_count", [0, 1, 4])
def test_loop_append_from_exact_empty_seed(model_runner, trip_count):
    model = _append_from_empty_model(trip_count=trip_count, width=3)
    x = np.arange(trip_count * 3, dtype=np.float32).reshape(trip_count, 3)
    actual, expected = model_runner.run_sample(
        model,
        [x],
        reference="cpu",
        name=f"loop_append_from_empty_seed_{trip_count}",
    )
    compare_outputs(actual, expected, atol=0.0, rtol=0.0)
