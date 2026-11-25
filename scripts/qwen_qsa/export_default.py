#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Stream a fresh, calibration-free Qwen default graph from verified BF16 weights.

The pinned release graph supplies topology only. Every learned constant is
regenerated from the official snapshot. No selector is loaded by this exporter.
"""

import argparse
import collections
import copy
import json
import re
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto as T
from onnx import helper as h
from safetensors import safe_open
from verify_handoff import ROOT, WORK, sha

TEMPLATE_SHA = "44b712df872250c1c1614c3aa717788a103f3a814e0cfd24b22196d3b99509cd"


def source_name(name):
    if name == "model.layers.64.final_norm_layernorm.weight":
        return "model.language_model.norm.weight"
    name = name.replace(".attn.", ".self_attn.").replace("_norm.layernorm", "_norm")
    return name.replace("model.layers.", "model.language_model.layers.")


class Export:
    def __init__(self, folder, snapshot):
        folder.mkdir(parents=True, exist_ok=False)
        self.folder, self.snapshot = folder, snapshot
        self.index = json.loads(
            (snapshot / "model.safetensors.index.json").read_text()
        )["weight_map"]
        self.data = (folder / "default.weights.bin").open("wb")
        self.initializers, self.conversion, self.used = [], {}, set()

    def read(self, name):
        self.used.add(name)
        with safe_open(self.snapshot / self.index[name], framework="pt") as stream:
            return stream.get_tensor(name)

    def constant(self, name, value):
        value = np.ascontiguousarray(value)
        self.data.write(b"\0" * ((-self.data.tell()) % 64))
        offset = self.data.tell()
        self.data.write(memoryview(value).cast("B"))
        tensor = T()
        tensor.name = name
        tensor.data_type = h.np_dtype_to_tensor_dtype(value.dtype)
        tensor.dims.extend(value.shape)
        tensor.data_location = T.EXTERNAL
        for key, val in [
            ("location", "default.weights.bin"),
            ("offset", offset),
            ("length", value.nbytes),
        ]:
            entry = tensor.external_data.add()
            entry.key, entry.value = key, str(val)
        self.initializers.append(tensor)
        return name

    def projection(self, node):
        name = node.input[1].split(".MatMul.")[0]
        source = source_name(name) + ".weight"
        weight = self.read(source)
        n, k = weight.shape
        attrs = {a.name: h.get_attribute_value(a) for a in node.attribute}
        assert (attrs["N"], attrs["K"]) == (n, k)
        keep_fp16 = name == "lm_head" or name.endswith((".in_proj_a", ".in_proj_b"))
        error2 = energy = 0.0
        maximum = 0.0
        underflow = 0
        if keep_fp16:
            # Transpose one matrix at a time; never instantiate the whole model.
            original = weight.float()
            converted = weight.half()
            assert torch.isfinite(converted).all()
            error = converted.float() - original
            error2, energy = float(error.square().sum()), float(original.square().sum())
            maximum = float(error.abs().max())
            underflow = int(((converted == 0) & (original != 0)).sum())
            name = self.constant(name + ".fresh_fp16", converted.numpy().T)
            result = h.make_node(
                "MatMul", [node.input[0], name], node.output, name=node.name
            )
            representation = "FP16"
        else:
            assert k % 128 == 0
            packed = np.empty((n, k // 128, 64), dtype=np.uint8)
            scales = np.empty((n, k // 128), dtype=np.float16)
            for start in range(0, n, 256):
                end = min(start + 256, n)
                original = (
                    weight[start:end].float().numpy().reshape(end - start, -1, 128)
                )
                scale = (np.abs(original).max(-1) / 7).astype(np.float16)
                scale[scale == 0] = np.float16(1)
                signed = np.clip(
                    np.rint(original / scale.astype(np.float32)[..., None]), -8, 7
                )
                codes = (signed.astype(np.int16) + 8).astype(np.uint8)
                packed[start:end] = codes[..., ::2] | (codes[..., 1::2] << 4)
                scales[start:end] = scale
                restored = signed * scale.astype(np.float32)[..., None]
                delta = restored - original
                error2 += float(np.square(delta, dtype=np.float64).sum())
                energy += float(np.square(original, dtype=np.float64).sum())
                maximum = max(maximum, float(np.abs(delta).max()))
            q = self.constant(name + ".fresh_rtn_q4", packed)
            s = self.constant(name + ".fresh_rtn_scales", scales)
            result = h.make_node(
                "MatMulNBits",
                [node.input[0], q, s],
                node.output,
                name=node.name,
                domain="com.microsoft",
                K=k,
                N=n,
                bits=4,
                block_size=128,
                accuracy_level=4,
            )
            representation = "RTN W4A16 group128, implicit zero point 8"
        self.conversion[source] = {
            "shape": [n, k],
            "representation": representation,
            "relative_rms": (error2 / energy) ** 0.5 if energy else 0.0,
            "max_abs_error": maximum,
            "underflow_to_zero": underflow,
        }
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, choices=[4, 64], default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    verification = json.loads((WORK / "handoff-verification.json").read_text())
    support = json.loads((WORK / "native-support.json").read_text())
    assert verification["backbone"]["weight_bytes_verified"]
    assert support["gate"]["executed_gate"] == "sigmoid"
    snapshot = Path(verification["backbone"]["snapshot"])
    template = ROOT.parent / "hip-ep-turboquant/models/qwen-original/text.onnx"
    assert sha(template) == TEMPLATE_SHA
    model = onnx.load(template, load_external_data=False)
    folder = args.output or WORK / f"default-rtn-w4a16-{args.layers}"
    exporter = Export(folder, snapshot)
    layer_re = re.compile(r"(?:layers\.|past_key_values\.|present\.)(\d+)")

    def retained(name):
        match = layer_re.search(name)
        return (
            not match or int(match[1]) < args.layers or "final_norm_layernorm" in name
        )

    nodes = []
    for node in model.graph.node:
        if not retained(node.name):
            continue
        if node.name == "/prune_logits/Gather":
            nodes.append(
                h.make_node(
                    "Gather",
                    [node.input[0], "logits_positions"],
                    ["/prune_logits/unsqueeze_out"],
                    axis=1,
                )
            )
            continue
        if node.name == "/prune_logits/Unsqueeze":
            continue
        if "final_norm_layernorm" in node.name and args.layers != 64:
            for i, name in enumerate(node.input):
                node.input[i] = name.replace("layers.63/", f"layers.{args.layers - 1}/")
        if node.op_type == "MatMulNBits":
            node = exporter.projection(node)
            print("Exported", len(exporter.conversion), node.name, flush=True)
        if node.op_type == "LinearAttention":
            # Existing runtime has one scalar dtype for both QKV and state.
            # Use its native FP32 region to retain the source FP32 recurrence.
            for i in [0, 1, 2, 4, 5]:
                original = node.input[i]
                converted = original + ".la_fp32"
                nodes.append(h.make_node("Cast", [original], [converted], to=T.FLOAT))
                node.input[i] = converted
            original = node.output[0]
            node.output[0] = original + ".la_fp32"
            nodes.append(node)
            nodes.append(
                h.make_node("Cast", [node.output[0]], [original], to=T.FLOAT16)
            )
            continue
        nodes.append(node)
    # Embed and decode in one full native session, with identical work in both variants.
    embedding = exporter.constant(
        "fresh.embed_tokens.weight",
        exporter.read("model.language_model.embed_tokens.weight").half().numpy(),
    )
    nodes.insert(
        0, h.make_node("Gather", [embedding, "input_ids"], ["inputs_embeds"], axis=0)
    )
    needed = {name for node in nodes for name in node.input}
    static = {
        "model.rotary_emb.cos_cache",
        "model.rotary_emb.sin_cache",
        "model.rotary_emb.h_mask",
        "model.rotary_emb.w_mask",
        "model.constants.q_scale",
        "prune_logits.last_index",
        "prune_logits.unsqueeze_axes",
    }
    for tensor in model.graph.initializer:
        name = tensor.name
        if name not in needed:
            continue
        if name in static:
            value = onnx.numpy_helper.to_array(tensor, base_dir=str(template.parent))
        elif name.endswith(".conv1d.bias"):
            value = np.zeros(tensor.dims, dtype=np.float16)
        else:
            source = source_name(name)
            if name.endswith(".neg_exp_A"):
                source = source.removesuffix(".neg_exp_A") + ".A_log"
                value = -exporter.read(source).float().exp().numpy()
            else:
                value = exporter.read(source).float()
                if "layernorm.weight" in name or "_norm.layernorm.weight" in name:
                    value = value + 1
                value = value.numpy().astype(
                    onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type)
                )
            assert list(value.shape) == list(tensor.dims), (
                name,
                value.shape,
                tensor.dims,
            )
        assert np.isfinite(value).all(), name
        exporter.constant(name, value)
    inputs = [
        x for x in model.graph.input if retained(x.name) and x.name != "inputs_embeds"
    ]
    inputs.insert(
        0,
        h.make_tensor_value_info(
            "input_ids", T.INT64, ["batch_size", "sequence_length"]
        ),
    )
    inputs.append(
        h.make_tensor_value_info("logits_positions", T.INT64, ["logits_count"])
    )
    outputs = [x for x in model.graph.output if retained(x.name)]
    for value in [*inputs, *outputs]:
        if value.name.endswith(".recurrent_state"):
            value.type.tensor_type.elem_type = T.FLOAT
        if value.name == "logits":
            value.type.tensor_type.shape.dim[1].ClearField("dim_value")
            value.type.tensor_type.shape.dim[1].dim_param = "logits_count"
    graph = h.make_graph(
        nodes, "Qwen fresh default", inputs, outputs, exporter.initializers
    )
    fresh = h.make_model(
        graph,
        opset_imports=list(model.opset_import),
        producer_name="hip-ep-qwen-fresh-default",
    )
    fresh.ir_version = model.ir_version
    exporter.data.close()
    path = folder / "default.onnx"
    onnx.save(fresh, path)
    # ORT's default-domain contrib norm has no stock ONNX schema. Check SSA,
    # ordering and all standard operators in a copy that marks it as contrib;
    # the actual graph retains its native domain and is validated by ORT next.
    structural = copy.deepcopy(fresh)
    for node in structural.graph.node:
        if node.op_type == "SimplifiedLayerNormalization" and not node.domain:
            node.domain = "com.microsoft"
    structural_path = folder / "structural-check.onnx"
    onnx.save(structural, structural_path)
    onnx.checker.check_model(str(structural_path))
    expected = {
        name
        for name in exporter.index
        if name == "lm_head.weight"
        or name == "model.language_model.embed_tokens.weight"
        or name == "model.language_model.norm.weight"
        or (name.startswith("model.language_model.layers.") and retained(name))
    }
    assert exporter.used == expected, (
        sorted(expected - exporter.used),
        sorted(exporter.used - expected),
    )
    receipt = {
        "status": "exported_not_validated",
        "structural_check": "passed; native contrib schema requires ORT session validation",
        "layers": args.layers,
        "model": verification["model"],
        "revision": verification["revision"],
        "template_topology_sha256": TEMPLATE_SHA,
        "learned_constants_source": "All regenerated from hash-verified official BF16 snapshot",
        "static_constants_source": "Pinned matching RoPE/shape constants; require native numerical validation",
        "source_parameters_used": sorted(exporter.used),
        "projections": exporter.conversion,
        "operators": dict(collections.Counter(n.op_type for n in nodes)),
        "recurrent_state": "FP32",
        "attention_gate": "native sigmoid",
        "training_updates": 0,
        "selectors_loaded": False,
        "onnx_sha256": sha(path),
        "data_bytes": Path(exporter.data.name).stat().st_size,
        "data_sha256": sha(Path(exporter.data.name)),
        "exporter_sha256": sha(Path(__file__)),
    }
    (folder / "export.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "folder": str(folder),
                "data_bytes": receipt["data_bytes"],
            }
        )
    )


if __name__ == "__main__":
    main()
