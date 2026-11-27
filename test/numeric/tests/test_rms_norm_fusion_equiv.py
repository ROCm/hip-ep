#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Basic numeric check for the RMSNorm expand pattern we fuse.
Builds the SD3.5-style primitive chain (fp32 stash Casts + Pow/ReduceMean/...
+ scale Mul) and checks ORT CPU against the closed-form RMSNorm. Does not
require hip-mlir-opt; fusion lowering is covered by the LIT file.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper


def _numpy_rms(x: np.ndarray, scale: np.ndarray, eps: float) -> np.ndarray:
    x32 = x.astype(np.float32)
    mean = np.mean(np.square(x32), axis=-1, keepdims=True)
    y32 = x32 / np.sqrt(mean + eps)
    return (y32.astype(np.float16) * scale).astype(np.float16)


def _expanded_onnx(shape, scale_shape, eps: float) -> bytes:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT16, list(shape))
    scale = helper.make_tensor_value_info(
        "scale", TensorProto.FLOAT16, list(scale_shape)
    )
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT16, list(shape))

    nodes = [
        helper.make_node(
            "Constant",
            [],
            ["two"],
            value=helper.make_tensor("two_t", TensorProto.FLOAT, [], [2.0]),
        ),
        helper.make_node(
            "Constant",
            [],
            ["eps"],
            value=helper.make_tensor("eps_t", TensorProto.FLOAT, [], [float(eps)]),
        ),
        helper.make_node(
            "Constant",
            [],
            ["one"],
            value=helper.make_tensor("one_t", TensorProto.FLOAT, [], [1.0]),
        ),
        helper.make_node(
            "Constant",
            [],
            ["axes"],
            value=helper.make_tensor("axes_t", TensorProto.INT64, [1], [-1]),
        ),
        helper.make_node("Cast", ["x"], ["x0"], to=TensorProto.FLOAT),
        helper.make_node("Pow", ["x0", "two"], ["sq"]),
        helper.make_node("ReduceMean", ["sq", "axes"], ["mean"], keepdims=1),
        helper.make_node("Add", ["mean", "eps"], ["added"]),
        helper.make_node("Sqrt", ["added"], ["rms"]),
        helper.make_node("Div", ["one", "rms"], ["inv"]),
        helper.make_node("Cast", ["x"], ["x1"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["x1", "inv"], ["norm"]),
        helper.make_node("Cast", ["norm"], ["norm_f16"], to=TensorProto.FLOAT16),
        helper.make_node("Mul", ["norm_f16", "scale"], ["y"]),
    ]
    model = helper.make_model(
        helper.make_graph(nodes, "rms_expanded", [x, scale], [y]),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    model.ir_version = 10
    return model.SerializeToString()


def test_expanded_chain_matches_numpy_rms():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 4, 8)).astype(np.float16)
    scale = rng.standard_normal((8,)).astype(np.float16)
    eps = 1e-6

    ref = _numpy_rms(x, scale, eps)
    sess = ort.InferenceSession(
        _expanded_onnx(x.shape, scale.shape, eps),
        providers=["CPUExecutionProvider"],
    )
    out = sess.run(None, {"x": x, "scale": scale})[0]
    max_abs = float(np.max(np.abs(out.astype(np.float32) - ref.astype(np.float32))))
    assert max_abs < 2e-3, f"max_abs_diff={max_abs}"


if __name__ == "__main__":
    test_expanded_chain_matches_numpy_rms()
    print("OK: expanded RMSNorm chain matches numpy reference")
