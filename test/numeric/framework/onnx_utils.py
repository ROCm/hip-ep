#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Generic ONNX utilities shared across all test files.

Provides op-agnostic helpers: type mapping and model wrapping/validation.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

NUMPY_TO_ONNX_TYPE = {
    np.float32: TensorProto.FLOAT,
    np.float64: TensorProto.DOUBLE,
    np.float16: TensorProto.FLOAT16,
    np.int64: TensorProto.INT64,
    np.int32: TensorProto.INT32,
    np.int16: TensorProto.INT16,
    np.uint16: TensorProto.UINT16,
    np.int8: TensorProto.INT8,
    np.uint8: TensorProto.UINT8,
    np.bool_: TensorProto.BOOL,
}

_ONNX_TO_NUMPY_TYPE = {v: k for k, v in NUMPY_TO_ONNX_TYPE.items()}


def np_to_onnx_type(dtype) -> int:
    """Map numpy dtype to ONNX TensorProto element type."""
    dtype = np.dtype(dtype)
    # ml_dtypes.bfloat16 has no numpy scalar type in the map above.
    if dtype.name == "bfloat16":
        return TensorProto.BFLOAT16
    return NUMPY_TO_ONNX_TYPE[dtype.type]


def onnx_type_to_np(onnx_type: int):
    """Map ONNX TensorProto element type to numpy dtype."""
    return np.dtype(_ONNX_TO_NUMPY_TYPE[onnx_type])


def fp64_normalize_last_axis(src: str, dst: str, epsilon: float, prefix: str = "norm"):
    """Nodes and initializers for (x - mean) / sqrt(var + epsilon) over the
    last axis of a DOUBLE tensor, as an exact two-pass computation.

    ORT CPU has no double InstanceNormalization, and its fp32 kernels round
    the mean of long rows with a large offset worse than a GPU tree reduction,
    so references for such inputs are built from these primitives.
    """
    p = prefix
    nodes = [
        helper.make_node("ReduceMean", [src], [f"{p}_mean"], axes=[-1], keepdims=1),
        helper.make_node("Sub", [src, f"{p}_mean"], [f"{p}_d"]),
        helper.make_node("Mul", [f"{p}_d", f"{p}_d"], [f"{p}_d2"]),
        helper.make_node(
            "ReduceMean", [f"{p}_d2"], [f"{p}_var"], axes=[-1], keepdims=1
        ),
        helper.make_node("Add", [f"{p}_var", f"{p}_eps"], [f"{p}_ve"]),
        helper.make_node("Sqrt", [f"{p}_ve"], [f"{p}_std"]),
        helper.make_node("Div", [f"{p}_d", f"{p}_std"], [dst]),
    ]
    inits = [numpy_helper.from_array(np.array(epsilon, np.float64), f"{p}_eps")]
    return nodes, inits


def make_model_from_nodes(
    nodes, inputs, outputs, initializers=None, opset=17, extra_opsets=None
):
    """Wrap nodes into a valid ONNX ModelProto with checker validation.

    For custom domain ops (e.g. com.microsoft), pass extra_opsets like
    [helper.make_opsetid("com.microsoft", 1)]. The ONNX checker is skipped
    when extra opsets are present since it cannot validate custom-domain
    operators.
    """
    graph = helper.make_graph(
        nodes, "test_graph", inputs, outputs, initializer=initializers or []
    )
    opset_imports = [helper.make_opsetid("", opset)]
    if extra_opsets:
        opset_imports.extend(extra_opsets)
    model = helper.make_model(graph, opset_imports=opset_imports)
    if extra_opsets is None:
        onnx.checker.check_model(model)
    return model
