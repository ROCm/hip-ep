#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""
Generate the tiny ONNX models used by test_e2e_migraphx_mock.cpp.

Elu has no hip-ep converter, so the HIP compiler claims none of these graphs
and gpu_backend=auto falls through to the MIGraphX eligibility rules.

- elu_two_io.onnx:  x0,x1 [1,4] -> y0 = Elu(x0 + x1), y1 = Elu(x0 - x1)
- elu_dynamic.onnx: x [N,4] -> Elu -> y            (symbolic input dim)
- elu_past.onnx:    past_key [1,4] -> Elu -> y     (KV-cache input name)
- lstm.onnx:        X [1,1,2] -> LSTM(hidden=1) -> Y_h [1,1,1]

Usage:
    python gen_migraphx_routing_models.py --output-dir models
"""

import argparse
import os

import numpy as np
import onnx
from onnx import TensorProto, helper


def save(graph, path):
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8
    )
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, path)


def elu_two_io():
    f = TensorProto.FLOAT
    return helper.make_graph(
        [
            helper.make_node("Add", ["x0", "x1"], ["sum"]),
            helper.make_node("Sub", ["x0", "x1"], ["diff"]),
            helper.make_node("Elu", ["sum"], ["y0"]),
            helper.make_node("Elu", ["diff"], ["y1"]),
        ],
        "elu_two_io",
        [
            helper.make_tensor_value_info("x0", f, [1, 4]),
            helper.make_tensor_value_info("x1", f, [1, 4]),
        ],
        [
            helper.make_tensor_value_info("y0", f, [1, 4]),
            helper.make_tensor_value_info("y1", f, [1, 4]),
        ],
    )


def elu_single(name, input_name, shape):
    f = TensorProto.FLOAT
    return helper.make_graph(
        [helper.make_node("Elu", [input_name], ["y"])],
        name,
        [helper.make_tensor_value_info(input_name, f, shape)],
        [helper.make_tensor_value_info("y", f, shape)],
    )


def lstm():
    f = TensorProto.FLOAT
    rng = np.random.default_rng(0)
    w = rng.standard_normal((1, 4, 2)).astype(np.float32)
    r = rng.standard_normal((1, 4, 1)).astype(np.float32)
    return helper.make_graph(
        [helper.make_node("LSTM", ["X", "W", "R"], ["", "Y_h"], hidden_size=1)],
        "lstm",
        [helper.make_tensor_value_info("X", f, [1, 1, 2])],
        [helper.make_tensor_value_info("Y_h", f, [1, 1, 1])],
        initializer=[
            helper.make_tensor("W", f, w.shape, w.flatten()),
            helper.make_tensor("R", f, r.shape, r.flatten()),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    out = args.output_dir
    save(elu_two_io(), os.path.join(out, "elu_two_io.onnx"))
    save(
        elu_single("elu_dynamic", "x", ["N", 4]), os.path.join(out, "elu_dynamic.onnx")
    )
    save(elu_single("elu_past", "past_key", [1, 4]), os.path.join(out, "elu_past.onnx"))
    save(lstm(), os.path.join(out, "lstm.onnx"))


if __name__ == "__main__":
    main()
