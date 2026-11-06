#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Add windowed hybrid-state contracts to an existing OGA text decoder."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import helper


WINDOWED_OPS = {
    "CausalConvWithState": (3, 1),
    "LinearAttention": (3, 1),
}


def _prepend_window(value: onnx.ValueInfoProto, window: int) -> None:
    shape = value.type.tensor_type.shape
    old_dimensions = []
    for dimension in shape.dim:
        copy = onnx.TensorShapeProto.Dimension()
        copy.CopyFrom(dimension)
        old_dimensions.append(copy)
    del shape.dim[:]
    shape.dim.add().dim_value = window
    shape.dim.extend(old_dimensions)


def add_state_window(input_path: Path, output_path: Path, window: int) -> int:
    if window <= 0:
        raise ValueError("state_window must be positive")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output must differ from input")

    model = onnx.load(input_path, load_external_data=False)
    graph_values = {
        value.name: value
        for value in (*model.graph.input, *model.graph.output, *model.graph.value_info)
    }
    state_names: set[str] = set()
    count = 0

    for node in model.graph.node:
        contract = WINDOWED_OPS.get(node.op_type)
        if node.domain != "com.microsoft" or contract is None:
            continue
        state_input_index, state_output_index = contract
        if (
            len(node.input) <= state_input_index
            or len(node.output) <= state_output_index
        ):
            raise ValueError(f"{node.name}: malformed {node.op_type} operands")
        if any(attribute.name == "state_window" for attribute in node.attribute):
            raise ValueError(f"{node.name}: state_window already exists")
        node.attribute.append(helper.make_attribute("state_window", window))
        state_names.add(node.input[state_input_index])
        state_names.add(node.output[state_output_index])
        count += 1

    if count == 0:
        raise ValueError("No windowable hybrid-state operators were found")

    missing = sorted(name for name in state_names if name not in graph_values)
    if missing:
        raise ValueError(
            "State tensors lack graph type information: " + ", ".join(missing)
        )
    for name in sorted(state_names):
        _prepend_window(graph_values[name], window)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, output_path)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--state-window", type=int, default=8)
    args = parser.parse_args()

    count = add_state_window(args.input, args.output, args.state_window)
    print(f"Updated {count} hybrid-state operators")
    print(f"Saved windowed decoder to {args.output}")


if __name__ == "__main__":
    main()
