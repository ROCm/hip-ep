#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""
Operator distribution of the compiler-input MLIR (step 1b).

Reads the graph the hip-ep init pass dumped -- the graph the compiler
actually sees, after ONNX Runtime's own optimizations -- and writes the same
JSON shape as step1_onnx_parser.py so the comparison and report builders can
consume either source.

Initializers become one `onnx.Constant` each in the MLIR form, and NoValue /
Return / EntryPoint are placeholders rather than graph nodes, so all of them
are excluded from the distribution. They are reported under
`_analysis_meta.excluded_carrier_ops`.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from mlir_text import (
    NON_COMPUTE_ONNX_OPS,
    element_type_names,
    parse_mlir_file,
    shape_kind,
    strip_quotes,
)


def analyze(mlir_path: Path, max_instances_per_op: int) -> dict:
    module = parse_mlir_file(mlir_path)

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

    for op in module.onnx_ops():
        op_type, domain = op.onnx_key()
        info = ops_info[op_type]
        info["count"] += 1
        if op.scope == "main_graph":
            info["count_top_level"] += 1
        info["domain"].add(domain)
        info["data_types"].update(element_type_names(op.type_signature))
        kind = shape_kind(op.type_signature)
        if kind:
            info["shape_types"].add(kind)
        info["scopes"].add(op.scope)

        if max_instances_per_op < 0 or len(info["instances"]) < max_instances_per_op:
            info["instances"].append(
                {
                    "node_name": strip_quotes(op.attrs.get("onnx_node_name", "")),
                    "graph_scope": op.scope,
                    "attributes": {k: op.attrs[k] for k in op.onnx_attr_names()},
                }
            )

    total = sum(info["count"] for info in ops_info.values())
    top_level = sum(info["count_top_level"] for info in ops_info.values())
    excluded = Counter(op.name for op in module.ops if op.name in NON_COMPUTE_ONNX_OPS)

    result = {
        "_analysis_meta": {
            "source": "compiler_input_mlir",
            "mlir_path": str(mlir_path),
            "include_subgraphs": True,
            "total_nodes": total,
            "top_level_nodes": top_level,
            "subgraph_nodes": total - top_level,
            "excluded_carrier_ops": dict(sorted(excluded.items())),
        }
    }
    for op_type, info in ops_info.items():
        result[op_type] = {
            "count": info["count"],
            "count_top_level": info["count_top_level"],
            "domain": sorted(info["domain"]),
            "data_types": sorted(info["data_types"]),
            "shape_types": sorted(info["shape_types"]),
            "scopes": sorted(info["scopes"])[:20],
            "instances": info["instances"],
        }
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mlir_path", help="compiler_input.mlir")
    ap.add_argument("output_dir", help="Directory for step1_onnx_ops.json")
    ap.add_argument(
        "--max-instances-per-op",
        type=int,
        default=5,
        metavar="N",
        help="Cap stored instances per op type (0=none, -1=unlimited; default 5)",
    )
    args = ap.parse_args()

    mlir_path = Path(args.mlir_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = analyze(mlir_path, args.max_instances_per_op)
    out_path = output_dir / "step1_onnx_ops.json"
    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    meta = result["_analysis_meta"]
    print(f"Analyzed {mlir_path}")
    print(f"  compute nodes: {meta['total_nodes']}")
    print(f"  excluded carriers: {meta['excluded_carrier_ops']}")
    print(f"[OK] JSON data saved: {out_path}")


if __name__ == "__main__":
    main()
