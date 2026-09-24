#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""
ONNX Model Analyzer - Step 1 (Optimized)
Analyze the operator distribution of an ONNX model (including ops inside
nested subgraphs such as Loop/If bodies).

Performance notes:
- Default uses onnx.load(..., load_external_data=False); we only need the
  graph and metadata, not the external weight files (large models keep
  weights in external .data files; loading them over UNC/network paths
  can be orders of magnitude slower).
- Default keeps at most max_instances_per_op entries per op type to avoid
  blowing up memory and JSON size on graphs with hundreds of thousands of
  nodes. One is enough downstream: without an EP input, the report reads
  the worklist's signatures and attributes from that instance.
- Default scope includes subgraphs (Loop.body, etc.). Use --top-level-only
  to restrict to the main graph (legacy behavior).
"""

import argparse
import onnx
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

from onnx_graph_walk import iter_model_nodes


class ONNXModelAnalyzer:
    def __init__(
        self,
        model_path: str,
        load_external_data: bool = False,
        include_subgraphs: bool = True,
    ):
        self.model_path = model_path
        self.include_subgraphs = include_subgraphs
        # load_external_data=False: do not load external weights into memory;
        # graph and initializer dtypes/dims are still readable.
        self.model = onnx.load(model_path, load_external_data=load_external_data)
        self.graph = self.model.graph

        # Pre-build a tensor info cache to avoid repeated lookups.
        self._tensor_cache = {}
        self._build_tensor_cache()

    def _build_tensor_cache(self):
        """Cache the type of every tensor the model states one for.

        value_info and subgraph scopes are included, not just the graph's own
        inputs, outputs and weights. Those three cover only the edges of the
        graph, so without the rest every intermediate tensor reads as
        unknown -- which is most of the model.
        """

        def record(value_info):
            tensor = value_info.type.tensor_type
            if not tensor.elem_type or value_info.name in self._tensor_cache:
                return
            self._tensor_cache[value_info.name] = {
                "dtype": self._get_dtype_name(tensor.elem_type),
                "shape": [
                    dim.dim_value if dim.dim_value > 0 else -1
                    for dim in tensor.shape.dim
                ],
                "shape_type": self._get_shape_type(
                    tensor.shape if tensor.HasField("shape") else None
                ),
            }

        def walk(graph):
            for initializer in graph.initializer:
                self._tensor_cache.setdefault(
                    initializer.name,
                    {
                        "dtype": self._get_dtype_name(initializer.data_type),
                        "shape": list(initializer.dims),
                        "shape_type": "static",
                    },
                )
            for collection in (graph.input, graph.output, graph.value_info):
                for value_info in collection:
                    record(value_info)
            for node in graph.node:
                for attr in node.attribute:
                    if attr.g.ByteSize():
                        walk(attr.g)
                    for sub_graph in attr.graphs:
                        walk(sub_graph)

        walk(self.graph)

    def _iter_nodes(self):
        if self.include_subgraphs:
            yield from iter_model_nodes(self.model)
        else:
            for node in self.graph.node:
                yield node, "main"

    def analyze(self, max_instances_per_op: Optional[int] = 5) -> Dict:
        """Analyze every operator in the model.

        max_instances_per_op:
            None - keep all instances (legacy behavior; very slow on large
                   models)
            0    - keep no instances (fastest; smallest JSON)
            >0   - keep at most this many per op_type (default 5; MD shows
                   only the first 3 so 5 is a comfortable margin)
        """
        ops_info = defaultdict(
            lambda: {
                "count": 0,
                "count_top_level": 0,
                "domain": set(),
                "data_types": set(),
                "shape_types": set(),
                "scopes": set(),
                "instances": [],
            }
        )

        top_level_total = len(self.graph.node)

        for node, scope in self._iter_nodes():
            op_type = node.op_type
            domain = node.domain if node.domain else "ai.onnx"

            # Collect input/output tensor info.
            input_info = []
            output_info = []

            for input_name in node.input:
                input_info.append(self._get_tensor_info(input_name))

            for output_name in node.output:
                output_info.append(self._get_tensor_info(output_name))

            # Extract data types and shape types from input + output tensors.
            data_types = set()
            shape_types = set()

            for tensor_info in input_info + output_info:
                if tensor_info["dtype"]:
                    data_types.add(tensor_info["dtype"])
                if tensor_info["shape_type"]:
                    shape_types.add(tensor_info["shape_type"])

            # Update statistics.
            ops_info[op_type]["count"] += 1
            if scope == "main":
                ops_info[op_type]["count_top_level"] += 1
            ops_info[op_type]["domain"].add(domain)
            ops_info[op_type]["data_types"].update(data_types)
            ops_info[op_type]["shape_types"].update(shape_types)
            ops_info[op_type]["scopes"].add(scope)

            inst = ops_info[op_type]["instances"]
            if max_instances_per_op is None or len(inst) < max_instances_per_op:
                inst.append(
                    {
                        "node_name": node.name,
                        "graph_scope": scope,
                        "inputs": input_info,
                        "outputs": output_info,
                        "attributes": {
                            attr.name: self._attr_to_str(attr)
                            for attr in node.attribute
                            if attr.type != onnx.AttributeProto.GRAPH
                            and attr.type != onnx.AttributeProto.GRAPHS
                        },
                    }
                )

        total_nodes = sum(info["count"] for info in ops_info.values())

        result = {
            "_analysis_meta": {
                "include_subgraphs": self.include_subgraphs,
                "total_nodes": total_nodes,
                "top_level_nodes": top_level_total,
                "subgraph_nodes": total_nodes - top_level_total
                if self.include_subgraphs
                else 0,
            }
        }
        for op_type, info in ops_info.items():
            result[op_type] = {
                "count": info["count"],
                "count_top_level": info["count_top_level"],
                # Sorted, like mlir_op_parser's: a set's iteration order
                # varies between runs, which would make two reports on the
                # same model differ in these columns for no reason.
                "domain": sorted(info["domain"]),
                "data_types": sorted(info["data_types"]),
                "shape_types": sorted(info["shape_types"]),
                "scopes": sorted(info["scopes"])[:20],
                "instances": info["instances"],
            }

        return result

    def _get_tensor_info(self, tensor_name: str) -> Dict:
        """Return the cached tensor info (no recursion)."""
        if tensor_name in self._tensor_cache:
            return {"name": tensor_name, **self._tensor_cache[tensor_name]}

        # Fall back to a default record when the tensor is unknown.
        return {
            "name": tensor_name,
            "dtype": None,
            "shape": [],
            "shape_type": "unknown",
        }

    def _get_dtype_name(self, dtype_int: int) -> str:
        """Name an ONNX TensorProto.DataType, the way numpy would.

        Asking the library rather than keeping a table: a hand-written one
        had uint32 through bfloat16 shifted by two positions, and reported
        every dtype added since as unknown.
        """
        if dtype_int == onnx.TensorProto.STRING:
            return "string"  # no numpy equivalent; "object" would not read as one
        try:
            return str(onnx.helper.tensor_dtype_to_np_dtype(dtype_int))
        except Exception:
            return onnx.TensorProto.DataType.Name(dtype_int).lower()

    def _get_shape_type(self, shape) -> str:
        """Classify a tensor shape as static / dynamic / unknown."""
        if not shape or not shape.dim:
            return "unknown"

        for dim in shape.dim:
            if (
                dim.dim_value <= 0
            ):  # Non-positive dim_value indicates a dynamic dimension.
                return "dynamic"

        return "static"

    def _attr_to_str(self, attr) -> str:
        """Render an ONNX attribute as a string."""
        if attr.HasField("f"):
            return str(attr.f)
        elif attr.HasField("i"):
            return str(attr.i)
        elif attr.HasField("s"):
            return attr.s.decode("utf-8")
        elif attr.floats:
            return str(list(attr.floats))
        elif attr.ints:
            return str(list(attr.ints))
        elif attr.strings:
            return str([s.decode("utf-8") for s in attr.strings])
        else:
            return "complex_value"


def main():
    ap = argparse.ArgumentParser(description="ONNX op distribution (step1)")
    ap.add_argument("model_path", help="Path to .onnx")
    ap.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="Output directory (default: next to model)",
    )
    ap.add_argument(
        "--max-instances-per-op",
        type=int,
        default=5,
        metavar="N",
        help="Cap stored instances per op type (0=none; default 5). Use -1 for unlimited.",
    )
    ap.add_argument(
        "--load-external-data",
        action="store_true",
        help="Load external weight files into memory (slow); default is graph-only.",
    )
    ap.add_argument(
        "--top-level-only",
        action="store_true",
        help="Count only main-graph nodes (legacy behavior; excludes Loop subgraph ops).",
    )
    args = ap.parse_args()

    model_path = args.model_path
    output_dir = args.output_dir or str(Path(model_path).parent)
    max_inst = args.max_instances_per_op
    if max_inst < 0:
        max_inst = None

    print(f"Analyzing ONNX model: {model_path}")
    print(f"Output directory: {output_dir}")
    print(
        f"Options: load_external_data={args.load_external_data}, "
        f"max_instances_per_op={max_inst if max_inst is not None else 'unlimited'}"
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    analyzer = ONNXModelAnalyzer(
        model_path,
        load_external_data=args.load_external_data,
        include_subgraphs=not args.top_level_only,
    )
    ops_info = analyzer.analyze(max_instances_per_op=max_inst)

    # Save the JSON data.
    json_path = Path(output_dir) / "step1_onnx_ops.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ops_info, f, indent=2, ensure_ascii=False)
    print(f"[OK] JSON data saved: {json_path}")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
