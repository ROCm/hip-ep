#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""
Operator distribution of the packaged ONNX model (step 1a).

Its only consumer is the comparison against the EP-input graph, which reads
the per-operator count. Everything the compatibility verdict needs comes from
the EP input instead, because that is the graph the compiler receives.

Nodes inside Loop / If / Scan bodies count too: they run, and ONNX Runtime
does not flatten them before the EP sees the graph.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import onnx


def iter_nodes(graph, scope="main"):
    """Yield (node, scope) for `graph` and every nested subgraph."""
    for node in graph.node:
        yield node, scope
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                yield from iter_nodes(attr.g, f"{scope}/{node.op_type}.{attr.name}")
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for index, sub_graph in enumerate(attr.graphs):
                    yield from iter_nodes(
                        sub_graph, f"{scope}/{node.op_type}.{attr.name}[{index}]"
                    )


def analyze(model_path: Path) -> dict:
    # Weights are irrelevant to a node count and a 16 GB external file is not
    # worth reading to get one.
    model = onnx.load(str(model_path), load_external_data=False)

    counts = defaultdict(int)
    domains = defaultdict(set)
    top_level = defaultdict(int)
    for node, scope in iter_nodes(model.graph):
        counts[node.op_type] += 1
        domains[node.op_type].add(node.domain or "ai.onnx")
        if scope == "main":
            top_level[node.op_type] += 1

    result = {
        "_analysis_meta": {
            "source": "original_onnx",
            "model_path": str(model_path),
            "total_nodes": sum(counts.values()),
            "top_level_nodes": sum(top_level.values()),
        }
    }
    for op_type, count in counts.items():
        result[op_type] = {
            "count": count,
            "count_top_level": top_level.get(op_type, 0),
            "domain": sorted(domains[op_type]),
        }
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_path", help="Path to .onnx")
    ap.add_argument("output_dir", help="Directory for step1_original_onnx_ops.json")
    args = ap.parse_args()

    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = analyze(model_path)
    out_path = output_dir / "step1_original_onnx_ops.json"
    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    meta = result["_analysis_meta"]
    print(f"Analyzed {model_path}")
    print(f"  nodes: {meta['total_nodes']} ({len(result) - 1} operator types)")
    print(f"[OK] {out_path}")


if __name__ == "__main__":
    main()
