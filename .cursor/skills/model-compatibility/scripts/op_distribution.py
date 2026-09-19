#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""
Operator counts for both graphs, and the difference between them (step 1).

The packaged ONNX and the EP input are the same model at two points: before
and after ONNX Runtime's graph optimizations. Counting them separately and
diffing them is one job, so it is one file; splitting it meant three scripts
and two intermediate files to express `a - b`.

Only the EP-input counts feed the compatibility verdict, because that is the
graph the compiler receives. The original is here to explain the difference:
a fused Swish becoming Sigmoid and Mul, a folded Constant, a dropped
Transpose.

Initializers are graph inputs in the protobuf but become one `onnx.Constant`
each in the MLIR form, so those carriers are excluded from the EP-input side
and reported under `_analysis_meta.excluded_carrier_ops`.
"""

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from mlir_text import NON_COMPUTE_ONNX_OPS, element_type_names, parse_mlir_file


def count_onnx_model(model_path: Path) -> dict:
    """Operator counts of the packaged ONNX, subgraphs included."""
    import onnx

    # Weights are irrelevant to a node count, and a 16 GB external file is not
    # worth reading to get one.
    model = onnx.load(str(model_path), load_external_data=False)

    def iter_nodes(graph, scope="main"):
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

    counts = defaultdict(int)
    top_level = defaultdict(int)
    domains = defaultdict(set)
    for node, scope in iter_nodes(model.graph):
        counts[node.op_type] += 1
        domains[node.op_type].add(node.domain or "ai.onnx")
        if scope == "main":
            top_level[node.op_type] += 1

    result = {
        "_analysis_meta": {
            "source": "original_onnx",
            "path": str(model_path),
            "total_nodes": sum(counts.values()),
            "top_level_nodes": sum(top_level.values()),
        }
    }
    for op_type, count in counts.items():
        result[op_type] = {
            "count": count,
            "count_top_level": top_level[op_type],
            "domain": sorted(domains[op_type]),
        }
    return result


def count_ep_input(mlir_path: Path) -> dict:
    """Operator counts of the EP-input MLIR."""
    module = parse_mlir_file(mlir_path)

    counts = defaultdict(int)
    top_level = defaultdict(int)
    domains = defaultdict(set)
    data_types = defaultdict(set)
    for op in module.onnx_ops():
        op_type, domain = op.onnx_key()
        counts[op_type] += 1
        domains[op_type].add(domain)
        data_types[op_type].update(element_type_names(op.type_signature))
        if op.scope == "main_graph":
            top_level[op_type] += 1

    excluded = Counter(op.name for op in module.ops if op.name in NON_COMPUTE_ONNX_OPS)
    result = {
        "_analysis_meta": {
            "source": "ep_input_mlir",
            "path": str(mlir_path),
            "total_nodes": sum(counts.values()),
            "top_level_nodes": sum(top_level.values()),
            "excluded_carrier_ops": dict(sorted(excluded.items())),
        }
    }
    for op_type, count in counts.items():
        result[op_type] = {
            "count": count,
            "count_top_level": top_level[op_type],
            "domain": sorted(domains[op_type]),
            "data_types": sorted(data_types[op_type]),
        }
    return result


def op_counts(distribution: dict):
    return {
        op_type: int(info.get("count", 0))
        for op_type, info in distribution.items()
        if not op_type.startswith("_") and isinstance(info, dict)
    }


def compare(original: dict, ep_input: dict, original_label: str, ep_label: str) -> dict:
    original_counts = op_counts(original)
    ep_counts = op_counts(ep_input)

    rows, only_original, only_ep, changed = [], [], [], []
    for op_type in sorted(set(original_counts) | set(ep_counts)):
        before = original_counts.get(op_type, 0)
        after = ep_counts.get(op_type, 0)
        row = {
            "op_type": op_type,
            "original_count": before,
            "ep_count": after,
            "delta": after - before,
        }
        rows.append(row)
        if before and not after:
            only_original.append(op_type)
        elif after and not before:
            only_ep.append(op_type)
        elif row["delta"]:
            changed.append(row)

    rows.sort(key=lambda r: (-max(r["original_count"], r["ep_count"]), r["op_type"]))

    return {
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "original_model": original_label,
            "ep_model": ep_label,
        },
        "summary": {
            "original_total_nodes": sum(original_counts.values()),
            "ep_total_nodes": sum(ep_counts.values()),
            "node_delta": sum(ep_counts.values()) - sum(original_counts.values()),
            "original_unique_ops": len(original_counts),
            "ep_unique_ops": len(ep_counts),
            "only_in_original": only_original,
            "only_in_ep": only_ep,
            "count_changed_ops": len(changed),
        },
        "rows": rows,
        "changed": changed,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_path", help="Packaged .onnx")
    ap.add_argument("output_dir", help="Directory for the distribution JSON files")
    ap.add_argument(
        "--ep-input",
        default="",
        help="ep_input.mlir; without it only the original is counted and no "
        "comparison is written, which is the -SkipDump case",
    )
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original = count_onnx_model(Path(args.model_path))
    write(output_dir / "step1_original_onnx_ops.json", original)
    print(f"  original ONNX: {original['_analysis_meta']['total_nodes']} nodes")

    if not args.ep_input:
        return

    ep_input = count_ep_input(Path(args.ep_input))
    write(output_dir / "step1_ep_input_ops.json", ep_input)
    meta = ep_input["_analysis_meta"]
    print(f"  EP input: {meta['total_nodes']} nodes")
    print(f"  excluded carriers: {meta['excluded_carrier_ops']}")

    comparison = compare(original, ep_input, args.model_path, args.ep_input)
    write(output_dir / "op_distribution_comparison.json", comparison)
    delta = comparison["summary"]["node_delta"]
    print(f"  delta: {delta:+d} nodes")


def write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] {path}")


if __name__ == "__main__":
    main()
