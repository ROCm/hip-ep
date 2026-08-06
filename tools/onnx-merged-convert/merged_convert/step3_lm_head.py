#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

_GATHER_NODE_NAME = "lm_head_gather"
_GATHER_INDEX_NAME = "lm_head_gather_index"
_UNSQUEEZE_NODE_NAME = "lm_head_unsqueeze"
_UNSQUEEZE_AXES_NAME = "lm_head_unsqueeze_axes"


def lm_head_has_gather_unsqueeze(model: onnx.ModelProto) -> bool:
    return any((n.name == _GATHER_NODE_NAME for n in model.graph.node))


def _gather_uses_const_last_index(graph: onnx.GraphProto) -> bool:
    for node in graph.node:
        if node.name != _GATHER_NODE_NAME or node.op_type != "Gather":
            continue
        if len(node.input) < 2 or node.input[1] != _GATHER_INDEX_NAME:
            return False
        for init in graph.initializer:
            if init.name == _GATHER_INDEX_NAME:
                return numpy_helper.to_array(init).item() == -1
        return False
    return False


def _migrate_gather_index_input_to_const(graph: onnx.GraphProto) -> bool:
    """Replace legacy ``head_token_index`` graph input with const ``-1`` initializer."""
    gather = next((n for n in graph.node if n.name == _GATHER_NODE_NAME), None)
    if gather is None or len(gather.input) < 2:
        return False
    index_name = gather.input[1]
    legacy_names = {"head_token_index", "head_head_token_index"}
    if index_name not in legacy_names and index_name != _GATHER_INDEX_NAME:
        return False
    if index_name == _GATHER_INDEX_NAME and _gather_uses_const_last_index(graph):
        return False
    gather.input[1] = _GATHER_INDEX_NAME
    if not any((init.name == _GATHER_INDEX_NAME for init in graph.initializer)):
        graph.initializer.append(
            numpy_helper.from_array(
                np.array(-1, dtype=np.int64), name=_GATHER_INDEX_NAME
            )
        )
    graph.input[:] = [vi for vi in graph.input if vi.name not in legacy_names]
    return True


def rewrite_lm_head_gather_unsqueeze(model: onnx.ModelProto) -> onnx.ModelProto:
    """Return *model* with Gather+Unsqueeze inserted before MatMulNBits."""
    if lm_head_has_gather_unsqueeze(model):
        if _migrate_gather_index_input_to_const(model.graph):
            onnx.checker.check_model(model)
        return model
    graph = model.graph
    mm_nodes = [n for n in graph.node if n.op_type == "MatMulNBits"]
    if len(mm_nodes) != 1:
        raise ValueError(
            f"Expected exactly one MatMulNBits node in lm_head, found {len(mm_nodes)}"
        )
    mm_node = mm_nodes[0]
    hidden_input = mm_node.input[0]
    logits_output = mm_node.output[0]
    gathered_name = "lm_head_gathered"
    mm_input_name = "lm_head_mm_input"
    graph.initializer.extend(
        [
            numpy_helper.from_array(
                np.array(-1, dtype=np.int64), name=_GATHER_INDEX_NAME
            ),
            numpy_helper.from_array(
                np.array([1], dtype=np.int64), name=_UNSQUEEZE_AXES_NAME
            ),
        ]
    )
    gather_node = helper.make_node(
        "Gather",
        [hidden_input, _GATHER_INDEX_NAME],
        [gathered_name],
        name=_GATHER_NODE_NAME,
        axis=1,
    )
    unsqueeze_node = helper.make_node(
        "Unsqueeze",
        [gathered_name, _UNSQUEEZE_AXES_NAME],
        [mm_input_name],
        name=_UNSQUEEZE_NODE_NAME,
    )
    mm_idx = next((i for i, n in enumerate(graph.node) if n is mm_node))
    graph.node.insert(mm_idx, gather_node)
    graph.node.insert(mm_idx + 1, unsqueeze_node)
    mm_node.input[0] = mm_input_name
    for vi in graph.output:
        if vi.name != logits_output:
            continue
        dims = vi.type.tensor_type.shape.dim
        if len(dims) >= 2:
            dims[1].ClearField("dim_param")
            dims[1].dim_value = 1
    onnx.checker.check_model(model)
    return model


def rewrite_lm_head_file(src: Path, dst: Path | None = None) -> Path:
    """Rewrite *src* and save to *dst* (default: ``<stem>_gather.onnx``)."""
    src = Path(src)
    dst = (
        Path(dst)
        if dst is not None
        else src.with_name(f"{src.stem}_gather{src.suffix}")
    )
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst
    model = onnx.load(str(src), load_external_data=True)
    model = rewrite_lm_head_gather_unsqueeze(model)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    onnx.save(model, str(dst))
    return dst
