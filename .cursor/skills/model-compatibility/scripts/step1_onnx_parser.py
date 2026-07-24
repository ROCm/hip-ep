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
  nodes. Downstream build_report_input only consumes per-op data_types,
  not the instance list.
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
    # Fallback descriptions, used ONLY when `onnx.defs.get_schema(...)` does
    # not return a schema. In practice this covers Microsoft custom ops
    # (com.microsoft domain) and any opset/domain combination the installed
    # `onnx` library does not know about. For standard ai.onnx ops the
    # description comes from the ONNX schema directly (single source of
    # truth: the ONNX specification, not this dict).
    OP_DESCRIPTIONS = {
        "MatMulNBits": "Quantized N-bit matrix multiplication (com.microsoft)",
        "RotaryEmbedding": "Rotary position embedding (RoPE)",
        "SkipSimplifiedLayerNormalization": "Skip connection + RMS normalization",
        "GroupQueryAttention": "Group Query Attention mechanism",
        "SimplifiedLayerNormalization": "RMS layer normalization",
        "MultiHeadAttention": "Multi-head attention (often inside Loop subgraph)",
        "CausalConvWithState": "Causal convolution with persistent state (com.microsoft)",
        "LinearAttention": "Linear attention (com.microsoft)",
        "QMoE": "Quantized Mixture-of-Experts (com.microsoft)",
    }

    # Domain -> canonical name accepted by onnx.defs.get_schema.
    _SCHEMA_DOMAIN_ALIASES = {
        "": "",
        "ai.onnx": "",
        "onnx": "",
        "com.microsoft": "com.microsoft",
    }

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
        self.op_descriptions = self._build_op_descriptions()

        # Pre-build a tensor info cache to avoid repeated lookups.
        self._tensor_cache = {}
        self._build_tensor_cache()

    def _build_tensor_cache(self):
        """Pre-build the tensor info cache for all initializers / inputs / outputs."""
        # Cache initializers.
        for initializer in self.graph.initializer:
            dtype = self._get_dtype_name(initializer.data_type)
            self._tensor_cache[initializer.name] = {
                "dtype": dtype,
                "shape": list(initializer.dims),
                "shape_type": "static",
            }

        # Cache graph inputs.
        for input_tensor in self.graph.input:
            dtype = self._get_dtype_name(input_tensor.type.tensor_type.elem_type)
            shape_type = self._get_shape_type(input_tensor.type.tensor_type.shape)
            shape = [
                dim.dim_value if dim.dim_value > 0 else -1
                for dim in input_tensor.type.tensor_type.shape.dim
            ]
            self._tensor_cache[input_tensor.name] = {
                "dtype": dtype,
                "shape": shape,
                "shape_type": shape_type,
            }

        # Cache graph outputs.
        for output_tensor in self.graph.output:
            dtype = self._get_dtype_name(output_tensor.type.tensor_type.elem_type)
            shape_type = self._get_shape_type(output_tensor.type.tensor_type.shape)
            shape = [
                dim.dim_value if dim.dim_value > 0 else -1
                for dim in output_tensor.type.tensor_type.shape.dim
            ]
            self._tensor_cache[output_tensor.name] = {
                "dtype": dtype,
                "shape": shape,
                "shape_type": shape_type,
            }

    def _build_op_descriptions(self) -> Dict[str, str]:
        """
        Build the operator description dictionary, sourced (in order) from:
          1. ONNX official op schema doc (`onnx.defs.get_schema(...).doc`)
             -- single source of truth for standard `ai.onnx` ops.
          2. The OP_DESCRIPTIONS fallback (mostly `com.microsoft` ops the
             installed `onnx` library may not carry, plus a few hand-curated
             one-liners).
          3. A synthesized `<OpType> (<domain>)` placeholder.
        """
        descriptions = {}

        for node, _scope in self._iter_nodes():
            op_type = node.op_type
            if op_type in descriptions:
                continue
            domain = node.domain if node.domain else "ai.onnx"
            doc = self._lookup_onnx_schema_doc(op_type, domain)
            if doc:
                descriptions[op_type] = doc
            elif op_type in self.OP_DESCRIPTIONS:
                descriptions[op_type] = self.OP_DESCRIPTIONS[op_type]
            else:
                descriptions[op_type] = f"{op_type} ({domain})"

        return descriptions

    def _lookup_onnx_schema_doc(self, op_type: str, domain: str) -> Optional[str]:
        """Return a short single-line description from the ONNX op schema.

        Trims the schema's `.doc` string to the first non-empty paragraph
        (or first sentence) to match the inline-friendly format used by
        the report's "Op Description" column.
        """
        schema_domain = self._SCHEMA_DOMAIN_ALIASES.get(domain.lower(), domain)
        try:
            schema = onnx.defs.get_schema(op_type, domain=schema_domain)
        except Exception:
            return None
        if schema is None or not getattr(schema, "doc", None):
            return None
        doc = schema.doc.strip()
        if not doc:
            return None
        # First non-empty paragraph (paragraphs separated by blank lines).
        para = doc.split("\n\n", 1)[0].strip()
        # Collapse internal whitespace / newlines for table-cell display.
        para = " ".join(para.split())
        # Prefer first sentence to keep the cell short; fall back to first
        # 200 chars if no sentence boundary is found.
        sentence_end = para.find(". ")
        if 0 < sentence_end <= 240:
            return para[: sentence_end + 1]
        return para[:240]

    def _iter_nodes(self):
        if self.include_subgraphs:
            yield from iter_model_nodes(self.model)
        else:
            for node in self.graph.node:
                yield node, "main"

    def get_op_description(self, op_type: str) -> str:
        """Return the operator description, or a default if not found."""
        return self.op_descriptions.get(op_type, f"{op_type} (unknown domain)")

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
                "domain": list(info["domain"]),
                "data_types": list(info["data_types"]),
                "shape_types": list(info["shape_types"]),
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
        """Map an ONNX TensorProto.DataType enum to a string name."""
        dtype_map = {
            1: "float32",
            2: "uint8",
            3: "int8",
            4: "uint16",
            5: "int16",
            6: "int32",
            7: "int64",
            10: "float16",
            11: "float64",
            12: "complex64",
            13: "complex128",
            14: "uint32",
            15: "uint64",
            16: "complex256",
        }
        return dtype_map.get(dtype_int, f"unknown({dtype_int})")

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


def generate_markdown_report(
    ops_info: Dict, model_name: str, analyzer: "ONNXModelAnalyzer"
) -> str:
    """Render the analysis as a Markdown report."""
    report = []
    meta = ops_info.get("_analysis_meta", {})
    op_items = {k: v for k, v in ops_info.items() if not k.startswith("_")}

    report.append(f"# ONNX Model Analysis Report: {model_name}\n")
    report.append("## Overview\n")

    total_ops = sum(info["count"] for info in op_items.values())
    unique_ops = len(op_items)

    report.append(f"- **Total Operators**: {total_ops}\n")
    if meta.get("include_subgraphs"):
        report.append(
            f"- **Top-level graph nodes only**: {meta.get('top_level_nodes', 'n/a')}\n"
        )
        report.append(
            f"- **Nodes inside subgraphs** (Loop/If/... bodies): {meta.get('subgraph_nodes', 0)}\n"
        )
    report.append(f"- **Unique Operator Types**: {unique_ops}\n")
    report.append(
        f"- **Scope**: {'main graph + nested subgraphs' if meta.get('include_subgraphs', True) else 'main graph only'}\n"
    )
    report.append(
        f"- **Analysis Date**: {__import__('datetime').datetime.now().isoformat()}\n\n"
    )

    report.append("## Shape Type Explanation\n\n")
    report.append(
        "- **static**: All dimensions have fixed values (e.g., [1, 128, 4096])\n"
    )
    report.append(
        "- **dynamic**: At least one dimension is variable (e.g., [batch_size, 128, 4096] where batch_size is -1)\n"
    )
    report.append(
        "- **unknown**: Shape information is not available in the graph (intermediate tensors without explicit shape definition)\n\n"
    )

    report.append("## Operator Distribution\n\n")
    report.append(
        "| Op Type | Count | Domain | Data Types | Shape Type | Description |\n"
    )
    report.append(
        "|---------|-------|--------|------------|------------|-------------|\n"
    )

    # Sort by count (descending).
    sorted_ops = sorted(op_items.items(), key=lambda x: x[1]["count"], reverse=True)

    for op_type, info in sorted_ops:
        count = info["count"]
        domain = ", ".join(info["domain"])
        data_types = ", ".join(info["data_types"]) if info["data_types"] else "-"
        shape_types = ", ".join(info["shape_types"]) if info["shape_types"] else "-"
        description = analyzer.get_op_description(op_type)

        report.append(
            f"| {op_type} | {count} | {domain} | {data_types} | {shape_types} | {description} |\n"
        )

    report.append("\n## Detailed Operator Information\n\n")

    for op_type, info in sorted_ops:
        report.append(f"### {op_type} (Count: {info['count']})\n\n")
        report.append(f"**Domain**: {', '.join(info['domain'])}\n\n")
        report.append(
            f"**Data Types**: {', '.join(info['data_types']) if info['data_types'] else 'Unknown'}\n\n"
        )
        report.append(
            f"**Shape Types**: {', '.join(info['shape_types']) if info['shape_types'] else 'Unknown'}\n\n"
        )

        total_n = info["count"]
        stored = info["instances"]
        if stored:
            sample_shown = min(3, len(stored))
            report.append(
                f"**Instances**: sample {sample_shown} of {len(stored)} recorded "
                f"(total nodes of this op type: {total_n})\n\n"
            )
            for i, instance in enumerate(stored[:3]):
                report.append(f"#### Instance {i + 1}: {instance['node_name']}\n\n")
                if instance["attributes"]:
                    report.append("**Attributes**:\n")
                    for attr_name, attr_value in instance["attributes"].items():
                        report.append(f"- {attr_name}: {attr_value}\n")
                    report.append("\n")

            remaining_in_graph = total_n - sample_shown
            if remaining_in_graph > 0:
                report.append(
                    f"... {remaining_in_graph} additional node(s) of this type in the graph "
                    f"(not all stored in JSON when capped)\n\n"
                )

    return "".join(report)


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

    # Derive a human-readable model name from the parent directory.
    model_name = Path(model_path).parent.name

    # Render the Markdown report.
    markdown_report = generate_markdown_report(ops_info, model_name, analyzer)

    # Save the Markdown report.
    md_path = Path(output_dir) / "step1_onnx_analysis.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_report)
    print(f"[OK] Markdown report saved: {md_path}")

    # Save the JSON data.
    json_path = Path(output_dir) / "step1_onnx_ops.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ops_info, f, indent=2, ensure_ascii=False)
    print(f"[OK] JSON data saved: {json_path}")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
