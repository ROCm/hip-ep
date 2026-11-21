#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Deterministically repair the v2 prototype graph without rewriting weights."""

import json

import onnx
from onnx import TensorProto as T
from onnx import helper as h
from prepare_baseline import original_folder, sha


def main():
    folder = original_folder()
    path = folder / "original.onnx"
    model = onnx.load(path, load_external_data=False)
    assert not any(n.op_type == "RotaryEmbedding" for n in model.graph.node)
    (folder / "original-mixed-rope-dtype.onnx").write_bytes(path.read_bytes())
    receipt = json.loads((folder / "export.json").read_text())
    (folder / "export-mixed-rope-dtype.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    weight_sha = sha(folder / "original.weights.bin")
    assert weight_sha == receipt["artifacts"]["original.weights.bin"]["sha256"]
    nodes = []
    for node in model.graph.node:
        if node.op_type == "GroupQueryAttention":
            layer = int(node.name.split(".")[2])
            kind = "full_attention" if layer % 4 == 3 else "sliding_attention"
            for index, label in [(0, "q"), (1, "k")]:
                stem = f"model.layers.{layer}.{label}"
                nodes.append(
                    h.make_node(
                        "Cast",
                        [node.input[index]],
                        [stem + "_rope_input_fp32"],
                        to=T.FLOAT,
                    )
                )
                nodes.append(
                    h.make_node(
                        "RotaryEmbedding",
                        [
                            stem + "_rope_input_fp32",
                            "position_ids",
                            "cos_" + kind,
                            "sin_" + kind,
                        ],
                        [stem + "_rope_fp32"],
                        domain="com.microsoft",
                        num_heads=32,
                        rotary_embedding_dim=128,
                        interleaved=0,
                    )
                )
                nodes.append(
                    h.make_node(
                        "Cast", [stem + "_rope_fp32"], [stem + "_rotated"], to=T.FLOAT16
                    )
                )
                node.input[index] = stem + "_rotated"
            node.input[7], node.input[8] = "", ""
            for attr in node.attribute:
                if attr.name == "do_rotary":
                    attr.i = 0
        nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.graph.input.append(
        h.make_tensor_value_info("position_ids", T.INT64, ["batch", "sequence"])
    )
    onnx.save(model, path)
    receipt["rotary_compute"] = "FP32 tables and FP32 rotation, then FP16 Q/K boundary"
    receipt["graph_repair"] = (
        "repair_rope_graph.py; packed backbone and other constants byte-identical"
    )
    receipt["artifacts"]["original.onnx"] = {
        "bytes": path.stat().st_size,
        "sha256": sha(path),
    }
    (folder / "export.json").write_text(json.dumps(receipt, indent=2) + "\n")
    validation = folder / "validation.json"
    if validation.exists():
        validation.rename(folder / "validation-mixed-rope-dtype-failure.json")
    print(
        "Repaired FP32 rotary graph; constants SHA256 unchanged", weight_sha, flush=True
    )


if __name__ == "__main__":
    main()
