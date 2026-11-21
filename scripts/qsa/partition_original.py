#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Export controlled Original attention boundaries without changing projections.

The nine ordered graphs alternate five unchanged backbone segments and four
single-node dense attention middles. Both comparison variants will share the
same backbone sessions; only the four explicit middle calls may be replaced.
"""

import hashlib
import json
import os

import onnx
from onnx import TensorProto as T
from onnx import helper as h
from prepare_baseline import original_folder, sha

LAYERS = [7, 11, 15, 23]


def spec(info):
    tensor = info.type.tensor_type
    return {
        "name": info.name,
        "dtype": tensor.elem_type,
        "shape": [
            d.dim_param if d.dim_param else d.dim_value for d in tensor.shape.dim
        ],
    }


def main():
    original = original_folder()
    assert json.loads((original / "validation.json").read_text())["status"] == "passed"
    folder = original / "partitioned"
    folder.mkdir(exist_ok=True)
    assert not (folder / "manifest.json").exists(), (
        "Preserve an existing partition export"
    )
    weights = original / "original.weights.bin"
    linked = folder / weights.name
    if not linked.exists():
        os.link(weights, linked)
    assert os.path.samefile(weights, linked)
    model = onnx.load(original / "original.onnx", load_external_data=False)
    types = {v.name: v for v in [*model.graph.input, *model.graph.output]}
    constants = {v.name: v for v in model.graph.initializer}
    groups, pending = [], []
    for node in model.graph.node:
        if (
            node.op_type == "GroupQueryAttention"
            and int(node.name.split(".")[2]) in LAYERS
        ):
            groups.append(("backbone", pending))
            groups.append(("attention", [node]))
            pending = []
        else:
            pending.append(node)
    groups.append(("backbone", pending))
    assert len(groups) == 9
    last_use = {}
    for index, (_, nodes) in enumerate(groups):
        for node in nodes:
            for name in node.input:
                if name:
                    last_use[name] = index
    for output in model.graph.output:
        last_use[output.name] = len(groups)
    manifest = {
        "original_graph_sha256": sha(original / "original.onnx"),
        "constants_sha256": sha(weights),
        "layers": LAYERS,
        "training_updates": 0,
        "projections": "All 224 source MatMulNBits nodes are byte-identical; shared backbone sessions in both variants",
        "parts": [],
        "boundaries": {},
        "last_use": last_use,
    }
    for index, (kind, nodes) in enumerate(groups):
        produced = {name for node in nodes for name in node.output if name}
        used = {name for node in nodes for name in node.input if name}
        incoming = sorted(used - produced - constants.keys())
        outgoing = sorted(name for name in produced if last_use.get(name, -1) > index)

        def info(name):
            return types.get(
                name,
                h.make_tensor_value_info(name, T.FLOAT16, ["batch", "sequence", 4096]),
            )

        graph = h.make_graph(
            nodes,
            f"Original_part_{index}_{kind}",
            [info(n) for n in incoming],
            [info(n) for n in outgoing],
            [constants[n] for n in sorted(used & constants.keys())],
        )
        part = h.make_model(
            graph, opset_imports=list(model.opset_import), ir_version=model.ir_version
        )
        filename = f"part-{index}-{kind}.onnx"
        onnx.save(part, folder / filename)
        record = {
            "file": filename,
            "kind": kind,
            "inputs": [spec(v) for v in graph.input],
            "outputs": [spec(v) for v in graph.output],
            "sha256": sha(folder / filename),
        }
        if kind == "attention":
            node = nodes[0]
            layer = int(node.name.split(".")[2])
            hidden = next(
                n.input[0]
                for n in model.graph.node
                if n.output[0] == f"model.layers.{layer}.self_attn.q_proj"
            )
            boundary = {
                "layer": layer,
                "part": index,
                "query": node.input[0],
                "key": node.input[1],
                "value": node.input[2],
                "hidden": hidden,
                "output": node.output[0],
                "past_key": node.input[3],
                "past_value": node.input[4],
                "present_key": node.output[1],
                "present_value": node.output[2],
                "precision": "FP16 normalized/rotated Q/K; FP16 V/KV/output",
            }
            manifest["boundaries"][str(layer)] = boundary
            record["layer"] = layer
        manifest["parts"].append(record)
    before = [
        n.SerializeToString() for n in model.graph.node if n.op_type == "MatMulNBits"
    ]
    after = [
        n.SerializeToString()
        for _, nodes in groups
        for n in nodes
        if n.op_type == "MatMulNBits"
    ]
    assert before == after and len(before) == 224
    manifest["projection_nodes_sha256"] = hashlib.sha256(b"".join(before)).hexdigest()
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        "Exported nine Original graphs; all backbone nodes and packed constants unchanged",
        flush=True,
    )


if __name__ == "__main__":
    main()
