#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Export all eight frozen selectors onto the preserved measured Qwen graph."""
# ruff: noqa: E402

import argparse
import hashlib
import json
import os
from pathlib import Path

import onnx
import torch
from onnx import TensorProto, helper
from verify_handoff import WORK, get_selector_asset, sha

TASK = WORK / "eight-qsa-v1"
LAYERS = [3, 7, 23, 27, 31, 35, 39, 63]
EXPORT_SHA = "8915ef00df9ad755808591862b473ee0e265cfd405ce1fc787bf948aea56eeeb"
CHECKPOINT_SHA = "76b1bcab417879a9d47d5cd8c023939ef35ada89d761c894c638461bd10a09b7"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def tensor_bytes(tensor):
    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=TASK / "models/Eight")
    args = parser.parse_args()
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    asset = get_selector_asset("qwen-eight")
    assert asset.spec["compatibility_sha256"] == EXPORT_SHA
    assert asset.config["layers"] == LAYERS
    assert asset.provenance["source_checkpoint_sha256"] == CHECKPOINT_SHA
    tensors = asset.tensors()
    source = WORK / "native-gqa/models/Original"
    exported = json.loads((source / "export.json").read_text())
    assert sha(source / "default.onnx") == exported["onnx_sha256"]
    assert sha(source / "default.weights.bin") == exported["data_sha256"]
    original = onnx.load(source / "default.onnx", load_external_data=False)
    graph = onnx.load(source / "default.onnx", load_external_data=False)
    folder = args.output_dir
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder / "default.onnx").exists(), "Preserve previous exports"
    os.link(source / "default.weights.bin", folder / "default.weights.bin")
    replacements = {
        f"/model/layers.{layer}/attn/GroupQueryAttention": layer for layer in LAYERS
    }
    shapes = {
        "projection.weight": (640, 5120),
        "q_norm.weight": (128,),
        "token_k_norm.weight": (128,),
        "pool_k_norm.weight": (128,),
    }
    hashes, digests, boundaries = {}, {}, {}
    graph.opset_import.append(helper.make_opsetid("com.amd", 1))
    with (folder / "qsa.selectors.bin").open("xb") as data:
        for node in graph.graph.node:
            if node.name not in replacements:
                continue
            layer = replacements[node.name]
            state = {name: tensors[f"selectors.{layer}.{name}"] for name in shapes}
            digest = hashlib.sha256()
            for name, tensor in state.items():
                assert tuple(tensor.shape) == shapes[name]
                assert tensor.dtype == torch.bfloat16 and torch.isfinite(tensor).all()
                digest.update(name.encode())
                digest.update(str(tuple(tensor.shape)).encode())
                digest.update(str(tensor.dtype).encode())
                digest.update(tensor_bytes(tensor))
            digests[str(layer)] = digest.hexdigest()
            assert (
                digests[str(layer)] == asset.spec["source_tensor_digests"][str(layer)]
            )
            assert (
                node.op_type == "GroupQueryAttention" and node.domain == "com.microsoft"
            )
            assert len(node.output) == 3
            assert node.input[3] == f"past_key_values.{layer}.key"
            q_proj = next(
                n
                for n in graph.graph.node
                if n.name == f"/model/layers.{layer}/attn/q_proj/MatMul_Q4"
            )
            hidden = q_proj.input[0]
            assert hidden == f"/model/layers.{layer}/input_layernorm/output_0"
            names = []
            for key, shape in shapes.items():
                name = f"selectors.{layer}.{key}"
                raw = tensor_bytes(state[key])
                offset = data.tell()
                data.write(raw)
                tensor = TensorProto(name=name, data_type=TensorProto.BFLOAT16)
                tensor.dims.extend(shape)
                tensor.data_location = TensorProto.EXTERNAL
                for key, value in [
                    ("location", "qsa.selectors.bin"),
                    ("offset", str(offset)),
                    ("length", str(len(raw))),
                ]:
                    tensor.external_data.add(key=key, value=value)
                graph.graph.initializer.append(tensor)
                names.append(name)
                hashes[name] = hashlib.sha256(raw).hexdigest()
            native = list(node.input[:7])
            boundaries[str(layer)] = dict(
                native_inputs=native, hidden=hidden, outputs=list(node.output)
            )
            node.op_type, node.domain = "QwenQsaGroupQueryAttention", "com.amd"
            del node.attribute[:]
            del node.input[:]
            node.input.extend(
                [
                    *native,
                    hidden,
                    *names,
                    "model.rotary_emb.cos_cache",
                    "model.rotary_emb.sin_cache",
                    "position_ids",
                ]
            )
    assert len(boundaries) == 8 and len(hashes) == 32
    assert len(graph.graph.node) == len(original.graph.node)
    assert all(
        a.SerializeToString() == b.SerializeToString()
        for a, b in zip(original.graph.node, graph.graph.node)
        if a.name not in replacements
    )
    assert all(
        a.SerializeToString() == b.SerializeToString()
        for a, b in zip(original.graph.initializer, graph.graph.initializer)
    )
    assert sum(n.op_type == "LinearAttention" for n in graph.graph.node) == 48
    assert sum(n.op_type == "QwenQsaGroupQueryAttention" for n in graph.graph.node) == 8
    assert sum(n.op_type == "GroupQueryAttention" for n in graph.graph.node) == 8
    onnx.save_model(graph, folder / "default.onnx")
    # Verify external bytes after serialization, independently of tensor writing.
    disk_graph = onnx.load(folder / "default.onnx", load_external_data=False)
    with (folder / "qsa.selectors.bin").open("rb") as data:
        for tensor in disk_graph.graph.initializer:
            if tensor.name not in hashes:
                continue
            info = {item.key: item.value for item in tensor.external_data}
            data.seek(int(info["offset"]))
            assert (
                hashlib.sha256(data.read(int(info["length"]))).hexdigest()
                == hashes[tensor.name]
            )
    receipt = dict(exported)
    receipt.update(
        variant="Changed",
        label="Eight",
        selectors_loaded=True,
        selector_layers=LAYERS,
        status="exported_not_validated",
        onnx_sha256=sha(folder / "default.onnx"),
        baseline_onnx_sha256=exported["onnx_sha256"],
        selector_sha256=EXPORT_SHA,
        selector_asset=asset.receipt(),
        source_checkpoint_sha256=CHECKPOINT_SHA,
        selector_tensor_sha256=hashes,
        source_tensor_digests=digests,
        selector_data_sha256=sha(folder / "qsa.selectors.bin"),
        boundaries=boundaries,
        exporter_sha256=sha(Path(__file__)),
        execution="one full ONNX graph/session; eight existing native Qwen QSA nodes",
        training_updates=0,
        learned_calibration_updates=0,
        backbone_constants_unchanged=True,
        phase_policy="sparse prefill and sparse decode",
    )
    save(folder / "export.json", receipt)
    save(folder / "selector-asset.json", asset.receipt())
    assert not torch.cuda.is_initialized()
    print(
        "Verified all 32 BF16 tensor payloads and eight per-layer digests; backbone unchanged",
        flush=True,
    )


if __name__ == "__main__":
    main()
