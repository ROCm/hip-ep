#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Add four frozen Qwen selectors without changing any default backbone node."""

import hashlib
import json
import os
from pathlib import Path

import onnx
from onnx import TensorProto, helper
from verify_handoff import PAYLOAD_SHA, SOURCE, WORK, header, sha


def main():
    source = WORK / "default-rtn-w4a16-64"
    exported = json.loads((source / "export.json").read_text())
    assert sha(source / "default.onnx") == exported["onnx_sha256"]
    manifest = json.loads(
        (SOURCE / "docs/reproducibility/qwen-hip-ep-manifest.json").read_text()
    )
    payload = SOURCE / manifest["portable_export"]["path"]
    assert sha(payload) == PAYLOAD_SHA
    descriptors, first = header(payload)
    original = onnx.load(source / "default.onnx", load_external_data=False)
    baseline_nodes = [node.SerializeToString() for node in original.graph.node]
    replacements = {
        f"/model/layers.{layer}/attn/GroupQueryAttention": layer
        for layer in [27, 31, 35, 39]
    }
    for variant in ["Original", "Changed"]:
        folder = WORK / "native-gqa/models" / variant
        folder.mkdir(parents=True, exist_ok=True)
        assert not (folder / "default.onnx").exists(), "Preserve previous exports"
        os.link(source / "default.weights.bin", folder / "default.weights.bin")
        graph = onnx.load(source / "default.onnx", load_external_data=False)
        hashes, boundaries = {}, {}
        if variant == "Changed":
            graph.opset_import.append(helper.make_opsetid("com.amd", 1))
            with (
                payload.open("rb") as stream,
                (folder / "qsa.selectors.bin").open("wb") as data,
            ):
                for node in graph.graph.node:
                    if node.name not in replacements:
                        continue
                    layer = replacements[node.name]
                    assert (
                        node.op_type == "GroupQueryAttention"
                        and node.domain == "com.microsoft"
                    )
                    assert (
                        len(node.output) == 3
                        and node.input[3] == f"past_key_values.{layer}.key"
                    )
                    q_proj = next(
                        n
                        for n in graph.graph.node
                        if n.name == f"/model/layers.{layer}/attn/q_proj/MatMul_Q4"
                    )
                    hidden = q_proj.input[0]
                    assert hidden == f"/model/layers.{layer}/input_layernorm/output_0"
                    names = []
                    for kind in ["projection", "q_norm", "token_k_norm", "pool_k_norm"]:
                        name = f"selectors.{layer}.{kind}.weight"
                        info = descriptors[name]
                        begin, end = info["data_offsets"]
                        stream.seek(first + begin)
                        raw = stream.read(end - begin)
                        digest = hashlib.sha256(raw).hexdigest()
                        assert (
                            digest
                            == manifest["tensors"][name]["raw_little_endian_sha256"]
                        )
                        offset = data.tell()
                        data.write(raw)
                        tensor = TensorProto(name=name, data_type=TensorProto.BFLOAT16)
                        tensor.dims.extend(info["shape"])
                        tensor.data_location = TensorProto.EXTERNAL
                        for key, value in [
                            ("location", "qsa.selectors.bin"),
                            ("offset", str(offset)),
                            ("length", str(len(raw))),
                        ]:
                            tensor.external_data.add(key=key, value=value)
                        graph.graph.initializer.append(tensor)
                        names.append(name)
                        hashes[name] = digest
                    native_inputs = list(node.input[:7])
                    boundaries[str(layer)] = {
                        "native_inputs": native_inputs,
                        "hidden": hidden,
                        "outputs": list(node.output),
                    }
                    node.op_type, node.domain = "QwenQsaGroupQueryAttention", "com.amd"
                    del node.attribute[:]
                    del node.input[:]
                    node.input.extend(
                        [
                            *native_inputs,
                            hidden,
                            *names,
                            "model.rotary_emb.cos_cache",
                            "model.rotary_emb.sin_cache",
                            "position_ids",
                        ]
                    )
            assert len(boundaries) == 4 and len(hashes) == 16
        changed = replacements if variant == "Changed" else {}
        assert len(graph.graph.node) == len(baseline_nodes)
        assert all(
            node.SerializeToString() == raw
            for node, raw in zip(graph.graph.node, baseline_nodes)
            if node.name not in changed
        )
        assert sum(n.op_type == "LinearAttention" for n in graph.graph.node) == 48
        assert sum(
            n.op_type == "QwenQsaGroupQueryAttention" for n in graph.graph.node
        ) == (4 if variant == "Changed" else 0)
        assert all(
            a.SerializeToString() == b.SerializeToString()
            for a, b in zip(original.graph.initializer, graph.graph.initializer)
        )
        onnx.save_model(graph, folder / "default.onnx")
        receipt = dict(exported)
        receipt.update(
            variant=variant,
            onnx_sha256=sha(folder / "default.onnx"),
            baseline_onnx_sha256=exported["onnx_sha256"],
            selector_sha256=PAYLOAD_SHA,
            selector_tensor_sha256=hashes,
            boundaries=boundaries,
            execution="one full ONNX graph/session",
            exporter_sha256=sha(Path(__file__)),
        )
        if variant == "Changed":
            receipt["selector_data_sha256"] = sha(folder / "qsa.selectors.bin")
        (folder / "exporter.py").write_bytes(Path(__file__).read_bytes())
        (folder / "export.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(variant, receipt["onnx_sha256"], flush=True)


if __name__ == "__main__":
    main()
