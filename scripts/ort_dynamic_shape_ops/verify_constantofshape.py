#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Verify ONNX Runtime `ConstantOfShape` output allocation on the CPU EP.

ConstantOfShape(shape) emits a tensor whose shape is literally the integer
*values* inside the rank-1 `shape` input, filled with the `value` attribute.
`shape` is fed as a runtime input (not an initializer), so static shape
inference knows only the output RANK (= len(shape)), never the dim sizes.
ORT must therefore allocate the output once it reads the shape values.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper

import common


def build_constantofshape_model() -> onnx.ModelProto:
    # rank-1 shape input of fixed length 2 -> output rank is 2, but the two
    # dim VALUES are unknown until runtime.
    shape_in = helper.make_tensor_value_info("shape", TensorProto.INT64, [2])
    y = helper.make_tensor_value_info(
        "y", TensorProto.FLOAT, ["d0", "d1"]
    )  # dims data-dependent
    fill = helper.make_tensor("value", TensorProto.FLOAT, [1], [7.0])
    node = helper.make_node("ConstantOfShape", ["shape"], ["y"], value=fill)
    graph = helper.make_graph([node], "cos_graph", [shape_in], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 10  # max IR version supported by onnxruntime 1.22
    onnx.checker.check_model(model)
    return model


def build_constantofshape_dynamic_rank_model() -> onnx.ModelProto:
    """Most-dynamic ConstantOfShape: even the output RANK is data-dependent.

    The `shape` input is declared rank-1 with a symbolic length ('rank'), so
    its element count -- and therefore the output's rank AND every dim -- is
    only known once the shape tensor is fed. The output is declared with
    unknown rank.
    """
    shape_in = helper.make_tensor_value_info("shape", TensorProto.INT64, ["rank"])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, None)  # unknown rank
    fill = helper.make_tensor("value", TensorProto.FLOAT, [1], [7.0])
    node = helper.make_node("ConstantOfShape", ["shape"], ["y"], value=fill)
    graph = helper.make_graph([node], "cos_dyn_graph", [shape_in], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 10
    # NOTE: no onnx.checker here. The checker requires every graph output to
    # carry a `shape`, but this output's RANK is data-dependent, so we leave
    # the shape absent (= unknown rank). ONNX Runtime accepts and executes
    # unknown-rank outputs; only the static checker objects.
    return model


def main() -> None:
    model = build_constantofshape_model()
    feeds = {"shape": np.array([2, 3], dtype=np.int64)}
    # Real output is 2x3 filled with 7.0. Probe exact, too-small, too-large.
    prealloc_cases = [
        ("correct 2x3", np.full((2, 3), -1.0, dtype=np.float32)),
        ("too small 2x2", np.full((2, 2), -1.0, dtype=np.float32)),
        ("too large 3x4", np.full((3, 4), -1.0, dtype=np.float32)),
    ]
    common.verify_op(
        title="ConstantOfShape  (output shape == the values inside `shape`)",
        description=(
            "feeds shape=[2,3], value=7.0  ->  expected 2x3 tensor of 7.0\n"
            "Static inference knows the output is rank-2 but not the 2 dim\n"
            "sizes; those come from the input data."
        ),
        model=model,
        feeds=feeds,
        prealloc_cases=prealloc_cases,
    )

    # Dynamic-shape input cases: `shape` has a symbolic length, so the output
    # rank itself varies per run. One session handles rank-1/2/3 outputs.
    dyn_cases = [
        ("shape=[2,3] -> rank-2", {"shape": np.array([2, 3], np.int64)}),
        ("shape=[4]   -> rank-1", {"shape": np.array([4], np.int64)}),
        ("shape=[2,2,2] -> rank-3", {"shape": np.array([2, 2, 2], np.int64)}),
        ("shape=[1,5] -> rank-2", {"shape": np.array([1, 5], np.int64)}),
    ]
    common.verify_dynamic_shapes(
        title="ConstantOfShape  (dynamic-shape input: output rank varies)",
        description=(
            "`shape` is declared rank-1 with a symbolic length, so the number\n"
            "of output dims is itself data-dependent. value=7.0 fills each."
        ),
        model=build_constantofshape_dynamic_rank_model(),
        cases=dyn_cases,
    )


if __name__ == "__main__":
    main()
