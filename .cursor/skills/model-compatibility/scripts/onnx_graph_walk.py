#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Traverse ONNX GraphProto nodes including nested subgraph attributes."""

from __future__ import annotations

from typing import Iterator, Tuple

import onnx
from onnx import GraphProto, NodeProto


def iter_graph_nodes(
    graph: GraphProto,
    scope: str = "main",
) -> Iterator[Tuple[NodeProto, str]]:
    """
    Yield (node, scope) for every node in `graph` and nested subgraphs.

    Nested graphs are reached via node attributes of type GRAPH or GRAPHS
    (e.g. Loop/If/Scan bodies, custom ops with embedded graphs).
    """
    for node in graph.node:
        yield node, scope
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                child = f"{scope}/{node.op_type}.{attr.name}"
                yield from iter_graph_nodes(attr.g, child)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for idx, sub_graph in enumerate(attr.graphs):
                    child = f"{scope}/{node.op_type}.{attr.name}[{idx}]"
                    yield from iter_graph_nodes(sub_graph, child)


def iter_model_nodes(model: onnx.ModelProto) -> Iterator[Tuple[NodeProto, str]]:
    """Yield all nodes from the model's top-level graph and nested subgraphs."""
    yield from iter_graph_nodes(model.graph)


def count_nodes_by_op(
    model: onnx.ModelProto,
    *,
    top_level_only: bool = False,
) -> Tuple[int, int]:
    """Return (total_node_count, unique_op_type_count)."""
    if top_level_only:
        nodes = model.graph.node
        return len(nodes), len({n.op_type for n in nodes})
    counts = set()
    total = 0
    for node, _ in iter_model_nodes(model):
        total += 1
        counts.add(node.op_type)
    return total, len(counts)
