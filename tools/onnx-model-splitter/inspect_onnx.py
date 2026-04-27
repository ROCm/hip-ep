#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""
Inspect ONNX model structure: list graph inputs, outputs, nodes,
initializers, and optionally dump node details.

Usage:
  python inspect_onnx.py <model.onnx>
  python inspect_onnx.py <model.onnx> --nodes              # list all nodes
  python inspect_onnx.py <model.onnx> --nodes --layer 0     # only show layer 0 nodes
  python inspect_onnx.py <model.onnx> --node-name "/model/layers.0/attn/q_proj/MatMul"
  python inspect_onnx.py <model.onnx> --nodes --type-name Add  # only show Add nodes
  python inspect_onnx.py <model.onnx> --op-types            # count by op type
  python inspect_onnx.py <model.onnx> --find-chains         # find common preprocessing chains
"""

import argparse
from collections import Counter

import onnx


def load_model(path):
    print(f"Loading {path} ...")
    m = onnx.load(path, load_external_data=False)
    print(f"  IR version:  {m.ir_version}")
    print(
        f"  Opset:       {[(o.domain or 'ai.onnx', o.version) for o in m.opset_import]}"
    )
    return m


def show_io(m):
    g = m.graph
    print(f"\nGraph: {g.name}")

    print(f"\nInputs ({len(g.input)}):")
    for inp in g.input:
        dims = []
        if inp.type.tensor_type.shape:
            for d in inp.type.tensor_type.shape.dim:
                if d.dim_param:
                    dims.append(d.dim_param)
                elif d.dim_value > 0:
                    dims.append(str(d.dim_value))
                else:
                    dims.append("?")
        elem = inp.type.tensor_type.elem_type
        print(f"  {inp.name:60s} dtype={elem:>2d}  shape=[{', '.join(dims)}]")

    print(f"\nOutputs ({len(g.output)}):")
    for out in g.output:
        dims = []
        if out.type.tensor_type.shape:
            for d in out.type.tensor_type.shape.dim:
                if d.dim_param:
                    dims.append(d.dim_param)
                elif d.dim_value > 0:
                    dims.append(str(d.dim_value))
                else:
                    dims.append("?")
        elem = out.type.tensor_type.elem_type
        print(f"  {out.name:60s} dtype={elem:>2d}  shape=[{', '.join(dims)}]")

    print(f"\nInitializers: {len(g.initializer)}")
    print(f"Nodes:        {len(g.node)}")

    detect_layers(g)


def detect_layers(g):
    """Detect transformer layers by scanning node names for layer index patterns."""
    import re

    pattern = re.compile(r"/layers?[./](\d+)[/.]")

    layer_node_counts = Counter()
    layer_sample_nodes = {}
    for n in g.node:
        match = pattern.search(n.name)
        if match:
            idx = int(match.group(1))
            layer_node_counts[idx] += 1
            if idx not in layer_sample_nodes:
                layer_sample_nodes[idx] = []
            layer_sample_nodes[idx].append((n.op_type, n.name))

    if not layer_node_counts:
        for n in g.node:
            match = re.search(r"_layer_?(\d+)_", n.name, re.IGNORECASE)
            if match:
                idx = int(match.group(1))
                layer_node_counts[idx] += 1
                if idx not in layer_sample_nodes:
                    layer_sample_nodes[idx] = []
                layer_sample_nodes[idx].append((n.op_type, n.name))

    if not layer_node_counts:
        print("Layers:       (not detected)")
        return

    all_indices = sorted(layer_node_counts.keys())
    min_l, max_l = all_indices[0], all_indices[-1]
    total = len(all_indices)
    contiguous = (max_l - min_l + 1) == total

    counts = [layer_node_counts[i] for i in all_indices]
    majority_count = Counter(counts).most_common(1)[0][0]
    normal_layers = [i for i in all_indices if layer_node_counts[i] == majority_count]
    outlier_layers = [i for i in all_indices if layer_node_counts[i] != majority_count]

    num_normal = len(normal_layers)
    print(
        f"Layers:       {num_normal}  (index {normal_layers[0]}..{normal_layers[-1]}, "
        f"{majority_count} nodes per layer"
        f"{'' if contiguous and not outlier_layers else ', see below'})"
    )

    if outlier_layers:
        print(
            f"              + {len(outlier_layers)} outlier(s) with different node count:"
        )
        for idx in outlier_layers:
            cnt = layer_node_counts[idx]
            nodes = layer_sample_nodes[idx]
            node_desc = ", ".join(f"{op}" for op, _ in nodes[:3])
            if len(nodes) > 3:
                node_desc += f" ... (+{len(nodes) - 3} more)"
            print(f"                layer {idx}: {cnt} node(s) - {node_desc}")
    elif not contiguous:
        missing = [i for i in range(min_l, max_l + 1) if i not in layer_node_counts]
        print(f"              missing indices: {missing}")


def show_nodes(m, layer_filter=None, type_name=None):
    g = m.graph
    type_lower = type_name.lower() if type_name else None
    matched = []
    for i, n in enumerate(g.node):
        if layer_filter is not None:
            if (
                f"/layers.{layer_filter}/" not in n.name
                and f"/layer.{layer_filter}/" not in n.name
            ):
                continue
        if type_lower is not None and n.op_type.lower() != type_lower:
            continue
        matched.append((i, n))

    label = (
        f"Nodes (showing {len(matched)}/{len(g.node)})"
        if (layer_filter is not None or type_name)
        else f"Nodes ({len(g.node)})"
    )
    print(f"\n{label}:")
    for i, n in matched:
        domain = f"[{n.domain}]" if n.domain else ""
        print(f"  {i:>5d}  {n.op_type:40s} {domain:25s} name={n.name}")
        for j, inp_name in enumerate(n.input):
            print(f"         in[{j}]: {inp_name}")
        for j, out_name in enumerate(n.output):
            print(f"         out[{j}]: {out_name}")


def show_node_detail(m, node_name):
    g = m.graph
    for n in g.node:
        if n.name == node_name:
            domain = f"[{n.domain}]" if n.domain else ""
            print(f"\nNode: {n.name}")
            print(f"  op_type: {n.op_type} {domain}")
            print(f"  inputs ({len(n.input)}):")
            for j, inp_name in enumerate(n.input):
                print(f"    [{j}] {inp_name}")
            print(f"  outputs ({len(n.output)}):")
            for j, out_name in enumerate(n.output):
                print(f"    [{j}] {out_name}")
            if n.attribute:
                print(f"  attributes ({len(n.attribute)}):")
                for a in n.attribute:
                    if a.type == onnx.AttributeProto.INT:
                        print(f"    {a.name} = {a.i}")
                    elif a.type == onnx.AttributeProto.FLOAT:
                        print(f"    {a.name} = {a.f}")
                    elif a.type == onnx.AttributeProto.STRING:
                        print(f"    {a.name} = {a.s.decode()}")
                    elif a.type == onnx.AttributeProto.INTS:
                        print(f"    {a.name} = {list(a.ints)}")
                    elif a.type == onnx.AttributeProto.FLOATS:
                        print(f"    {a.name} = {list(a.floats)}")
                    else:
                        print(f"    {a.name} (type={a.type})")
            return
    print(f"  Node '{node_name}' not found")


def show_op_types(m):
    g = m.graph
    counter = Counter(n.op_type for n in g.node)
    print(f"\nOp type counts ({len(counter)} types, {len(g.node)} total nodes):")
    for op, cnt in counter.most_common():
        print(f"  {op:40s} {cnt:>5d}")


def find_chains(m):
    """Find common preprocessing chains that might need deletion."""
    g = m.graph
    out_to_node = {}
    for n in g.node:
        for o in n.output:
            out_to_node[o] = n

    print("\n--- Chains starting from graph inputs ---")
    for inp in g.input:
        consumers = [n for n in g.node if inp.name in n.input]
        for c in consumers:
            if c.op_type in ("Shape", "Gather", "Reshape", "Cast"):
                chain = [f"{inp.name} ->"]
                current = c
                for _ in range(10):
                    chain.append(f"{current.op_type}({current.name})")
                    next_consumers = [
                        n2
                        for n2 in g.node
                        if any(o in n2.input for o in current.output)
                        and n2.op_type
                        in (
                            "Shape",
                            "Gather",
                            "Cast",
                            "Unsqueeze",
                            "Concat",
                            "Reshape",
                            "Sub",
                            "Add",
                        )
                    ]
                    if len(next_consumers) == 1:
                        chain.append(" -> ")
                        current = next_consumers[0]
                    else:
                        break
                print("  " + "".join(chain))


def main():
    parser = argparse.ArgumentParser(description="Inspect ONNX model structure")
    parser.add_argument("model", type=str, help="Path to .onnx file")
    parser.add_argument("--nodes", action="store_true", help="List all nodes")
    parser.add_argument(
        "--layer", type=int, default=None, help="Filter nodes by layer index"
    )
    parser.add_argument(
        "--node-name", type=str, default=None, help="Show detail for a specific node"
    )
    parser.add_argument(
        "--type-name",
        type=str,
        default=None,
        help="Filter --nodes by op_type (case-insensitive), e.g. --type-name Add",
    )
    parser.add_argument(
        "--op-types", action="store_true", help="Count nodes by op type"
    )
    parser.add_argument(
        "--find-chains",
        action="store_true",
        help="Find preprocessing chains from inputs",
    )
    args = parser.parse_args()

    m = load_model(args.model)
    show_io(m)

    if args.op_types:
        show_op_types(m)
    if args.type_name and not args.nodes:
        args.nodes = True
    if args.nodes:
        show_nodes(m, layer_filter=args.layer, type_name=args.type_name)
    if args.node_name:
        show_node_detail(m, args.node_name)
    if args.find_chains:
        find_chains(m)

    if not (args.nodes or args.node_name or args.op_types or args.find_chains):
        print(
            "\nTip: use --nodes, --op-types, --node-name, or --find-chains for more details"
        )


if __name__ == "__main__":
    main()
