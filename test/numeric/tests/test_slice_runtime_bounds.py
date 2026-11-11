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
_I32 = TensorProto.INT32
_I64 = TensorProto.INT64
_BOOL = TensorProto.BOOL


def _loop_model(body_nodes, body_initializers, row_shape, trip=3):
    """Loop over ``x[8,4]`` that scans the body's ``row`` output.

    ``x_in`` is loop-carried without a declared shape, as in a TF export, so
    ORT cannot type a Slice of it and the importer leaves the result unranked.
    """
    x = helper.make_tensor_value_info("x", _F32, [8, 4])
    body = helper.make_graph(
        [
            helper.make_node("Identity", ["cond_in"], ["cond_out"]),
            helper.make_node("Identity", ["x_in"], ["x_out"]),
            *body_nodes,
        ],
        "body",
        [
            helper.make_tensor_value_info("iter", _I64, []),
            helper.make_tensor_value_info("cond_in", _BOOL, []),
            helper.make_tensor_value_info("x_in", _F32, None),
        ],
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
            helper.make_tensor_value_info("scan", _F32, [None, *row_shape]),
        ],
        initializers=[trip_init, cond_init],
    )


def _loop_same_bounds_model():
    """``Slice(x_in, b, b, axes=[0])``: one runtime value for both bounds.

    The slice is empty at compile time, so it lowers to a bare tensor.empty
    with no hip.slice. ONNX leaves the mean of an empty set undefined: ORT
    CPU gives 0, so only the scan shape is compared.
    """
    initializers = [
        numpy_helper.from_array(np.array([1], dtype=np.int64), "vec1"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "axis0"),
    ]
    nodes = [
        helper.make_node("Reshape", ["iter", "vec1"], ["bound"]),
        helper.make_node("Slice", ["x_in", "bound", "bound", "axis0"], ["empty"]),
        helper.make_node("ReduceMean", ["empty"], ["row"], axes=[0], keepdims=0),
    ]
    return _loop_model(nodes, initializers, [4])


def _loop_strided_tail_model():
    """Every second column from ``-3`` to a clamped runtime end, i32 bounds.

    Axis 1 is sliced, not the leading one. The start is negative, the end
    ``iter + 10`` is past the 4 columns and clamps, and the step is 2, so each
    iteration keeps columns 1 and 3. The scan row is the mean of those two
    columns for each of the 8 rows.
    """
    initializers = [
        numpy_helper.from_array(np.array(10, dtype=np.int64), "ten"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "vec1"),
        numpy_helper.from_array(np.array([-3], dtype=np.int32), "start"),
        numpy_helper.from_array(np.array([1], dtype=np.int32), "axis1"),
        numpy_helper.from_array(np.array([2], dtype=np.int32), "step2"),
    ]
    nodes = [
        helper.make_node("Add", ["iter", "ten"], ["end_scalar"]),
        helper.make_node("Reshape", ["end_scalar", "vec1"], ["end_i64"]),
        helper.make_node("Cast", ["end_i64"], ["end"], to=_I32),
        helper.make_node("Slice", ["x_in", "start", "end", "axis1", "step2"], ["cols"]),
        helper.make_node("ReduceMean", ["cols"], ["row"], axes=[1], keepdims=0),
    ]
    return _loop_model(nodes, initializers, [8])


def _loop_prefix_model(trip: int, *, offset: int = 1):
    """Scan row i is the mean of the first i+offset rows of ``x[8,4]``.

    With ``offset=0`` the first iteration's slice is empty, and ORT gives 0
    for its mean.
    """
    initializers = [
        numpy_helper.from_array(np.array(offset, dtype=np.int64), "one"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "vec1"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "zero_vec"),
    ]
    nodes = [
        helper.make_node("Add", ["iter", "one"], ["end_scalar"]),
        helper.make_node("Reshape", ["end_scalar", "vec1"], ["end_vec"]),
        helper.make_node(
            "Slice", ["x_in", "zero_vec", "end_vec", "zero_vec"], ["prefix"]
        ),
        helper.make_node("ReduceMean", ["prefix"], ["row"], axes=[0], keepdims=0),
    ]
    return _loop_model(nodes, initializers, [4], trip=trip)


class TestSliceRuntimeBounds:
    def test_loop_body_prefix(self, model_runner):
        model = _loop_prefix_model(3)
        x = np.arange(1, 33, dtype=np.float32).reshape(8, 4)
        actual, expected = model_runner.run_sample(model, [x])
        assert actual[1].shape == (3, 4)
        compare_outputs(actual, expected, atol=1e-5, rtol=1e-5)

    def test_loop_body_empty_prefix(self, model_runner):
        model = _loop_prefix_model(3, offset=0)
        x = np.arange(1, 33, dtype=np.float32).reshape(8, 4)
        actual, expected = model_runner.run_sample(model, [x])
        assert actual[1].shape == (3, 4)
        compare_outputs(actual, expected, atol=1e-5, rtol=1e-5)

    def test_loop_body_same_bounds(self, model_runner):
        model = _loop_same_bounds_model()
        x = np.arange(1, 33, dtype=np.float32).reshape(8, 4)
        actual, expected = model_runner.run_sample(model, [x])
        assert actual[1].shape == expected[1].shape == (3, 4)
        compare_outputs(actual[:1], expected[:1], atol=1e-5, rtol=1e-5)

    def test_loop_body_strided_tail(self, model_runner):
        model = _loop_strided_tail_model()
        x = np.arange(1, 33, dtype=np.float32).reshape(8, 4)
        actual, expected = model_runner.run_sample(model, [x])
        assert actual[1].shape == (3, 8)
        compare_outputs(actual, expected, atol=1e-5, rtol=1e-5)
