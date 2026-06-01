#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Verify ONNX Runtime `Range` output allocation on the CPU EP.

Range(start, limit, delta) emits a 1-D tensor whose LENGTH is
ceil((limit - start) / delta) -- a function of the input *values*, not of
any input shape. start/limit/delta are passed as runtime scalar inputs
(not initializers) so the output length is genuinely unknown until the
op executes. Static shape inference therefore reports an unknown ('?')
length, and ORT has to allocate the right-sized buffer at run time.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper

import common


def build_range_model() -> onnx.ModelProto:
    start = helper.make_tensor_value_info("start", TensorProto.INT64, [])
    limit = helper.make_tensor_value_info("limit", TensorProto.INT64, [])
    delta = helper.make_tensor_value_info("delta", TensorProto.INT64, [])
    # Output declared rank-1 with an unknown ('?') length -- data dependent.
    y = helper.make_tensor_value_info("y", TensorProto.INT64, ["range_len"])
    node = helper.make_node("Range", ["start", "limit", "delta"], ["y"])
    graph = helper.make_graph([node], "range_graph", [start, limit, delta], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10  # max IR version supported by onnxruntime 1.22
    onnx.checker.check_model(model)
    return model


def build_range_from_dynamic_input_model() -> onnx.ModelProto:
    """Range driven by a DYNAMIC-shape input -- the real-world idiom.

    Range's own operands are scalars (rank-0), so there is no input *shape*
    to vary directly. Instead we feed a tensor `data` of shape
    [batch_size, sequence_length] (both symbolic) and compute
    Range(0, Shape(data)[1], 1) -- i.e. position ids [0 .. seq_len). The
    output length is bound to the dynamic `sequence_length` dim at run time,
    which is exactly how transformer exports build position_ids.
    """
    data = helper.make_tensor_value_info(
        "data", TensorProto.FLOAT, ["batch_size", "sequence_length"]
    )
    positions = helper.make_tensor_value_info(
        "positions", TensorProto.INT64, ["sequence_length"]
    )

    def scalar_const(out, val):
        return helper.make_node(
            "Constant",
            [],
            [out],
            value=helper.make_tensor(out + "_t", TensorProto.INT64, [], [val]),
        )

    nodes = [
        helper.make_node("Shape", ["data"], ["shp"]),  # -> [2] = [batch, seq]
        scalar_const("idx1", 1),
        helper.make_node("Gather", ["shp", "idx1"], ["seqlen"], axis=0),  # 0-D seq
        scalar_const("start", 0),
        scalar_const("delta", 1),
        helper.make_node("Range", ["start", "seqlen", "delta"], ["positions"]),
    ]
    graph = helper.make_graph(nodes, "range_dyn_graph", [data], [positions])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def main() -> None:
    model = build_range_model()
    feeds = {
        "start": np.array(0, dtype=np.int64),
        "limit": np.array(10, dtype=np.int64),
        "delta": np.array(2, dtype=np.int64),
    }
    # The real output length is 5 ([0 2 4 6 8]). Probe pre-bound buffers of
    # the exact length, too small, and too large to see how ORT reacts.
    prealloc_cases = [
        ("correct len 5", np.full(5, -1, dtype=np.int64)),
        ("too small len 3", np.full(3, -1, dtype=np.int64)),
        ("too large len 8", np.full(8, -1, dtype=np.int64)),
    ]
    common.verify_op(
        title="Range  (output length depends on start/limit/delta values)",
        description=(
            "feeds start=0 limit=10 delta=2  ->  expected [0 2 4 6 8], length 5\n"
            "The length is not encoded in any input shape, so it cannot be\n"
            "pre-computed from shapes alone."
        ),
        model=model,
        feeds=feeds,
        prealloc_cases=prealloc_cases,
    )

    # Dynamic-shape input cases: Range(0, Shape(data)[1], 1) where `data` has
    # symbolic [batch_size, sequence_length]. One session, many seq lengths.
    dyn_cases = [
        ("data 1x5  -> positions[0:5]", {"data": np.zeros((1, 5), np.float32)}),
        ("data 2x8  -> positions[0:8]", {"data": np.zeros((2, 8), np.float32)}),
        ("data 1x1  -> positions[0:1]", {"data": np.zeros((1, 1), np.float32)}),
        ("data 3x12 -> positions[0:12]", {"data": np.zeros((3, 12), np.float32)}),
    ]
    common.verify_dynamic_shapes(
        title="Range  (dynamic-shape input: position-ids idiom)",
        description=(
            "Range operands are scalars, so we drive it from a dynamic input:\n"
            "positions = Range(0, Shape(data)[1], 1).  `data` is declared\n"
            "[batch_size, sequence_length] (symbolic); the output length tracks\n"
            "the runtime sequence_length."
        ),
        model=build_range_from_dynamic_input_model(),
        cases=dyn_cases,
    )


if __name__ == "__main__":
    main()
