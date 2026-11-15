#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Execute onnx.Loop scan outputs and compare values and shapes.

The lit tests check the lowered IR. These runs check the tensor the loop
actually returns: a finished trip count, a zero-trip loop, a condition that
stops the loop early, and two scan outputs from one loop.

Each scan row is the carried input plus the iteration index. The carried
inputs themselves pass through unchanged, so a zero-trip loop returns them
as-is and an empty scan.
"""

from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

_F32 = TensorProto.FLOAT
_I64 = TensorProto.INT64
_BOOL = TensorProto.BOOL


def _compare_scans(actual, expected):
    assert len(actual) == len(expected)
    for index, (got, want) in enumerate(zip(actual, expected)):
        assert got.shape == want.shape, (
            f"output {index}: shape {got.shape} vs {want.shape}"
        )
        # compare_outputs calls max() on the difference, which rejects an
        # empty zero-trip scan. Matching shapes are enough for that output.
        if got.size == 0:
            continue
        compare_outputs([got], [want], atol=1e-5, rtol=1e-5)


def _scan_model(trip: int, *, early: bool = False, second: bool = False):
    """Build a Loop whose scan row i is the carried input plus i.

    Outputs are the carried finals, then the scans. When ``early`` is set,
    the body condition is ``iter < 1``, so iterations 0 and 1 run and
    iteration 2 does not, even if ``trip`` is larger.
    """
    x = helper.make_tensor_value_info("x", _F32, [4])
    graph_inputs = [x]
    loop_inputs = ["trip", "cond", "x"]
    body_inputs = [
        helper.make_tensor_value_info("iter", _I64, []),
        helper.make_tensor_value_info("cond_in", _BOOL, []),
        helper.make_tensor_value_info("x_in", _F32, [4]),
    ]
    body_initializers = []
    if early:
        body_initializers.append(
            numpy_helper.from_array(np.array(1, dtype=np.int64), "limit")
        )
        cond_node = helper.make_node("Less", ["iter", "limit"], ["cond_out"])
    else:
        cond_node = helper.make_node("Identity", ["cond_in"], ["cond_out"])

    carried_nodes = [helper.make_node("Identity", ["x_in"], ["x_out"])]
    carried_outputs = [helper.make_tensor_value_info("x_out", _F32, [4])]
    scan_nodes = [
        helper.make_node("Cast", ["iter"], ["iter_f"], to=_F32),
        helper.make_node("Add", ["x_in", "iter_f"], ["chunk"]),
    ]
    scan_outputs = [helper.make_tensor_value_info("chunk", _F32, [4])]
    loop_outputs = ["x_final", "scan"]
    graph_outputs = [
        helper.make_tensor_value_info("x_final", _F32, [4]),
        helper.make_tensor_value_info("scan", _F32, [None, 4]),
    ]
    if second:
        graph_inputs.append(helper.make_tensor_value_info("y", _F32, [4]))
        loop_inputs.append("y")
        body_inputs.append(helper.make_tensor_value_info("y_in", _F32, [4]))
        carried_nodes.append(helper.make_node("Identity", ["y_in"], ["y_out"]))
        carried_outputs.append(helper.make_tensor_value_info("y_out", _F32, [4]))
        scan_nodes.append(helper.make_node("Add", ["y_in", "iter_f"], ["chunk_b"]))
        scan_outputs.append(helper.make_tensor_value_info("chunk_b", _F32, [4]))
        loop_outputs = ["x_final", "y_final", "scan", "scan_b"]
        graph_outputs = [
            helper.make_tensor_value_info("x_final", _F32, [4]),
            helper.make_tensor_value_info("y_final", _F32, [4]),
            helper.make_tensor_value_info("scan", _F32, [None, 4]),
            helper.make_tensor_value_info("scan_b", _F32, [None, 4]),
        ]

    body = helper.make_graph(
        [cond_node, *carried_nodes, *scan_nodes],
        "body",
        body_inputs,
        [
            helper.make_tensor_value_info("cond_out", _BOOL, []),
            *carried_outputs,
            *scan_outputs,
        ],
        initializer=body_initializers,
    )
    trip_init = numpy_helper.from_array(np.array(trip, dtype=np.int64), "trip")
    cond_init = numpy_helper.from_array(np.array(True), "cond")
    node = helper.make_node("Loop", loop_inputs, loop_outputs, body=body)
    return make_model_from_nodes(
        [node],
        graph_inputs,
        graph_outputs,
        initializers=[trip_init, cond_init],
    )


def _body_sized_scan_model(trip: int):
    """Scan whose row length is computed in the body: Concat(x_in, x_in).

    ``x_in`` is loop-carried without a declared shape, so the row length is
    not known before the loop. For a zero-trip loop ORT gives such a scan
    shape (0, 0); a static row length would be kept instead.
    """
    body = helper.make_graph(
        [
            helper.make_node("Identity", ["cond_in"], ["cond_out"]),
            helper.make_node("Identity", ["x_in"], ["x_out"]),
            helper.make_node("Concat", ["x_in", "x_in"], ["chunk"], axis=0),
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
            helper.make_tensor_value_info("chunk", _F32, [None]),
        ],
    )
    trip_init = numpy_helper.from_array(np.array(trip, dtype=np.int64), "trip")
    cond_init = numpy_helper.from_array(np.array(True), "cond")
    node = helper.make_node(
        "Loop", ["trip", "cond", "x"], ["x_final", "scan"], body=body
    )
    return make_model_from_nodes(
        [node],
        [helper.make_tensor_value_info("x", _F32, [4])],
        [
            helper.make_tensor_value_info("x_final", _F32, [4]),
            helper.make_tensor_value_info("scan", _F32, [None, None]),
        ],
        initializers=[trip_init, cond_init],
    )


class TestLoopScan:
    def test_full_trip(self, model_runner):
        model = _scan_model(3)
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        actual, expected = model_runner.run_sample(model, [x])
        _compare_scans(actual, expected)
        assert actual[1].shape == (3, 4)

    def test_zero_trip(self, model_runner):
        model = _scan_model(0)
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        actual, expected = model_runner.run_sample(model, [x])
        _compare_scans(actual, expected)
        assert actual[1].shape == (0, 4)

    def test_early_stop(self, model_runner):
        model = _scan_model(5, early=True)
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        actual, expected = model_runner.run_sample(model, [x])
        _compare_scans(actual, expected)
        assert actual[1].shape == (2, 4)

    def test_two_scan_outputs(self, model_runner):
        model = _scan_model(3, second=True)
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        y = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        actual, expected = model_runner.run_sample(model, [x, y])
        _compare_scans(actual, expected)
        assert actual[2].shape == (3, 4)
        assert actual[3].shape == (3, 4)

    def test_zero_trip_body_sized(self, model_runner):
        model = _body_sized_scan_model(0)
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        actual, expected = model_runner.run_sample(model, [x])
        _compare_scans(actual, expected)
        assert actual[1].shape == (0, 0)

    def test_full_trip_body_sized(self, model_runner):
        model = _body_sized_scan_model(2)
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        actual, expected = model_runner.run_sample(model, [x])
        _compare_scans(actual, expected)
        assert actual[1].shape == (2, 8)
