#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Verify ONNX Runtime `NonZero` output allocation on the CPU EP.

NonZero(x) returns the row-major indices of the non-zero elements of x as
an int64 tensor of shape [rank(x), N], where N = number of non-zero
elements. N is the most thoroughly data-dependent size of the three ops:
it depends on the actual *data* in x, not even on a shape-like input.
Static inference resolves the first dim (= rank) but leaves N unknown, so
ORT must size the output buffer after counting the non-zeros at run time.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper

import common


def build_nonzero_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [3, 4])
    # Output is [2, N]: 2 == rank(x) (static), N is data-dependent ('?').
    y = helper.make_tensor_value_info("y", TensorProto.INT64, [2, "num_nonzero"])
    node = helper.make_node("NonZero", ["x"], ["y"])
    graph = helper.make_graph([node], "nonzero_graph", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10  # max IR version supported by onnxruntime 1.22
    onnx.checker.check_model(model)
    return model


def build_nonzero_dynamic_model() -> onnx.ModelProto:
    """NonZero with symbolic input dims [rows, cols].

    Both the input shape AND the output column count N vary at run time, so a
    single session must size each result after counting non-zeros.
    """
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["rows", "cols"])
    y = helper.make_tensor_value_info("y", TensorProto.INT64, [2, "num_nonzero"])
    node = helper.make_node("NonZero", ["x"], ["y"])
    graph = helper.make_graph([node], "nonzero_dyn_graph", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def main() -> None:
    model = build_nonzero_model()
    # 5 non-zero elements -> output is [2, 5].
    x = np.array(
        [
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 3.0, 0.0],
            [4.0, 0.0, 0.0, 5.0],
        ],
        dtype=np.float32,
    )
    feeds = {"x": x}
    prealloc_cases = [
        ("correct 2x5", np.full((2, 5), -1, dtype=np.int64)),
        ("too small 2x3", np.full((2, 3), -1, dtype=np.int64)),
        ("too large 2x12", np.full((2, 12), -1, dtype=np.int64)),
    ]
    common.verify_op(
        title="NonZero  (output column count N == #non-zero elements of x)",
        description=(
            "feeds x = 3x4 with 5 non-zeros  ->  expected [2, 5] index tensor\n"
            "N depends purely on the data values of x; no input shape encodes\n"
            "it. This is the canonical 'count then allocate' case."
        ),
        model=model,
        feeds=feeds,
        prealloc_cases=prealloc_cases,
    )

    # Dynamic-shape input cases: x has symbolic [rows, cols]. One session,
    # different input shapes AND different non-zero counts (incl. the N=0
    # empty-output edge case).
    dyn_cases = [
        (
            "3x4, 5 non-zeros",
            {
                "x": np.array(
                    [[0, 1, 0, 2], [0, 0, 3, 0], [4, 0, 0, 5]], dtype=np.float32
                )
            },
        ),
        ("2x2, all zeros (N=0)", {"x": np.zeros((2, 2), dtype=np.float32)}),
        ("1x6, 4 non-zeros", {"x": np.array([[1, 0, 2, 3, 0, 4]], dtype=np.float32)}),
        (
            "5x1, 2 non-zeros",
            {"x": np.array([[0], [7], [0], [0], [9]], dtype=np.float32)},
        ),
    ]
    common.verify_dynamic_shapes(
        title="NonZero  (dynamic-shape input: symbolic [rows, cols])",
        description=(
            "x is declared [rows, cols] (both symbolic). Each run feeds a\n"
            "different shape and a different number of non-zeros; ORT sizes\n"
            "the [2, N] output per run (N can be 0)."
        ),
        model=build_nonzero_dynamic_model(),
        cases=dyn_cases,
    )


if __name__ == "__main__":
    main()
