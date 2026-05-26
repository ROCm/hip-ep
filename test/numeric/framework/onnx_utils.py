#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Generic ONNX utilities shared across all test files.

Provides op-agnostic helpers: type mapping and model wrapping/validation.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper

NUMPY_TO_ONNX_TYPE = {
    np.float32: TensorProto.FLOAT,
    np.float64: TensorProto.DOUBLE,
    np.float16: TensorProto.FLOAT16,
    np.int64: TensorProto.INT64,
    np.int32: TensorProto.INT32,
    np.int16: TensorProto.INT16,
    np.int8: TensorProto.INT8,
    np.uint8: TensorProto.UINT8,
    np.bool_: TensorProto.BOOL,
}

_ONNX_TO_NUMPY_TYPE = {v: k for k, v in NUMPY_TO_ONNX_TYPE.items()}


def np_to_onnx_type(dtype) -> int:
    """Map numpy dtype to ONNX TensorProto element type."""
    dtype = np.dtype(dtype)
    return NUMPY_TO_ONNX_TYPE[dtype.type]


def onnx_type_to_np(onnx_type: int):
    """Map ONNX TensorProto element type to numpy dtype."""
    return np.dtype(_ONNX_TO_NUMPY_TYPE[onnx_type])


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


def bind_dims_partial(model, dim_map):
    """Rebind selected ``dim_param`` slots in ``model`` to concrete values.

    Unlike the historical "freeze every dim" helpers used by the static
    embedding tests (e.g. ``_bind_num_logical_patches``), this helper
    only rewrites dims whose ``dim_param`` is a key in ``dim_map`` and
    leaves every other dim_param untouched. Dim slots with ``dim_value``
    set are also preserved.

    Parameters
    ----------
    model : onnx.ModelProto
        The input model. NOT mutated -- a deep copy is returned so
        repeated calls with different ``dim_map``s share the original.
    dim_map : Mapping[str, int]
        ``{ dim_param_name: concrete_value }`` for each dim that should
        be pinned. Missing entries stay symbolic.

    Returns
    -------
    onnx.ModelProto
        A deep copy of ``model`` with the requested dims rebound. The
        ONNX checker is not re-run because partial-rebind graphs are
        intentionally still dynamic, which is the whole point of the
        helper.

    Notes
    -----
    - Walks both ``graph.input`` and ``graph.output`` (matching the
      pattern used in the static fixtures), so an output-only dim_param
      with no corresponding input slot is still rebound.
    - Skips ``value_info`` deliberately: the MorphiZen importer
      re-derives intermediate shapes, and rebinding them here would
      create spurious mismatches if the user's ``dim_map`` doesn't
      cover every internal symbol.
    """
    import copy

    out = copy.deepcopy(model)
    for tv in list(out.graph.input) + list(out.graph.output):
        for d in tv.type.tensor_type.shape.dim:
            if d.HasField("dim_param") and d.dim_param in dim_map:
                d.Clear()
                d.dim_value = int(dim_map[d.dim_param])
    return out
