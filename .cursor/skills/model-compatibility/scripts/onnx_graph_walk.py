#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Traverse ONNX GraphProto nodes including nested subgraph attributes."""

from __future__ import annotations

from typing import Iterator, NamedTuple, Tuple

import onnx
from onnx import GraphProto, NodeProto, ValueInfoProto, helper


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


class NodeContext(NamedTuple):
    """A node together with everything needed to rebuild it on its own.

    `types` is the whole visible scope, not just this graph: an ONNX subgraph
    may reference values defined in an enclosing one, and those references
    have to become inputs of a standalone model.
    """

    node: NodeProto
    scope: str
    types: dict[str, ValueInfoProto]
    constants: dict[str, NodeProto]
    produced_here: frozenset[str]


def _scope_types(graph: GraphProto) -> dict[str, ValueInfoProto]:
    out: dict[str, ValueInfoProto] = {}
    for coll in (graph.input, graph.output, graph.value_info):
        for v in coll:
            out[v.name] = v
    for init in graph.initializer:
        out.setdefault(
            init.name,
            helper.make_tensor_value_info(init.name, init.data_type, list(init.dims)),
        )
    return out


def iter_typed_nodes(
    graph: GraphProto,
    scope: str = "main",
    outer_types: dict[str, ValueInfoProto] | None = None,
    outer_constants: dict[str, NodeProto] | None = None,
) -> Iterator[NodeContext]:
    """Yield every node with its visible types and in-scope Constant nodes.

    Run shape inference on the model first; this reads value_info rather
    than deriving anything.
    """
    types = dict(outer_types or {})
    types.update(_scope_types(graph))
    constants = dict(outer_constants or {})
    for n in graph.node:
        if n.op_type == "Constant" and n.output:
            constants[n.output[0]] = n
    produced = frozenset(o for n in graph.node for o in n.output if o)

    for node in graph.node:
        yield NodeContext(node, scope, types, constants, produced)
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                child = f"{scope}/{node.op_type}.{attr.name}"
                yield from iter_typed_nodes(attr.g, child, types, constants)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for idx, sub_graph in enumerate(attr.graphs):
                    child = f"{scope}/{node.op_type}.{attr.name}[{idx}]"
                    yield from iter_typed_nodes(sub_graph, child, types, constants)


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
