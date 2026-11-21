#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Deterministic, calibration-free OLMo3 RTN W4A16 export for hip-ep.

Uses the original 224 separate backbone projections and an FP16 output head.
Q/K normalization is over 4096 channels, before GQA splits heads. YaRN tables
come from the pinned model config and the installed Transformers implementation.
No module is trained and no dataset is consulted by this exporter.
"""

import argparse
import json

import numpy as np
import onnx
import torch
from onnx import TensorProto as T
from onnx import helper as h
from prepare_baseline import REFERENCE, REVISION, WORK, original_folder, sha
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.olmo3.modeling_olmo3 import Olmo3RotaryEmbedding


def quantize(weight, group=128, bits=4):
    """FP16 absmax scale, ties-to-even RTN and implicit midpoint zero point."""
    w = weight.float().numpy()
    n, k = w.shape
    assert k % group == 0
    blocks = w.reshape(n, k // group, group)
    assert bits in [4, 8]
    midpoint = 1 << (bits - 1)
    scales = (np.abs(blocks).max(-1) / (midpoint - 1)).astype(np.float16)
    scales[scales == 0] = np.float16(1)
    codes = np.clip(
        np.rint(blocks / scales.astype(np.float32)[..., None]), -midpoint, midpoint - 1
    )
    codes = (codes.astype(np.int16) + midpoint).astype(np.uint8)
    packed = codes[..., ::2] | (codes[..., 1::2] << 4) if bits == 4 else codes
    return packed, scales


class Export:
    def __init__(self, folder, layers, bits=4):
        self.folder, self.layers, self.bits = folder, layers, bits
        folder.mkdir(parents=True, exist_ok=True)
        assert not (folder / "original.onnx").exists(), (
            "Preserve the existing export and its hard-linked constants"
        )
        self.weights = json.loads(
            (WORK / "base-model/model.safetensors.index.json").read_text()
        )["weight_map"]
        self.nodes, self.initializers = [], []
        self.data = (folder / "original.weights.bin").open("wb")
        self.quantization = {}

    def source(self, name):
        with safe_open(WORK / "base-model" / self.weights[name], framework="pt") as f:
            return f.get_tensor(name)

    def constant(self, name, value):
        value = np.ascontiguousarray(value)
        # Write raw arrays incrementally; the protobuf never contains large weights.
        padding = (-self.data.tell()) % 64
        self.data.write(b"\0" * padding)
        offset = self.data.tell()
        self.data.write(value.tobytes())
        tp = T()
        tp.name = name
        tp.data_type = h.np_dtype_to_tensor_dtype(value.dtype)
        tp.dims.extend(value.shape)
        tp.data_location = T.EXTERNAL
        for key, val in [
            ("location", "original.weights.bin"),
            ("offset", offset),
            ("length", value.nbytes),
        ]:
            entry = tp.external_data.add()
            entry.key, entry.value = key, str(val)
        self.initializers.append(tp)
        return name

    def node(self, op, inputs, output, **attrs):
        outputs = [output] if isinstance(output, str) else output
        self.nodes.append(h.make_node(op, inputs, outputs, name=outputs[0], **attrs))
        return outputs[0]

    def norm(self, x, name):
        scale = self.constant(
            name + ".weight", self.source(name + ".weight").half().numpy()
        )
        return self.node(
            "SimplifiedLayerNormalization",
            [x, scale],
            name,
            axis=-1,
            epsilon=1e-6,
            stash_type=1,
        )

    def linear(self, x, name, quantized=True):
        weight = self.source(name + ".weight")
        n, k = weight.shape
        if quantized:
            packed, scales = quantize(weight, bits=self.bits)
            b = self.constant(name + ".packed", packed)
            s = self.constant(name + ".scales", scales)
            self.quantization[name] = {
                "K": k,
                "N": n,
                "bits": self.bits,
                "block_size": 128,
                "zero_point": 1 << (self.bits - 1),
                "packed_sha256": __import__("hashlib")
                .sha256(packed.tobytes())
                .hexdigest(),
            }
            return self.node(
                "MatMulNBits",
                [x, b, s],
                name,
                domain="com.microsoft",
                K=k,
                N=n,
                bits=self.bits,
                block_size=128,
                accuracy_level=0,
            )
        b = self.constant(
            name + ".weight_transposed", weight.half().t().contiguous().numpy()
        )
        return self.node("MatMul", [x, b], name)

    def build(self):
        assert __import__("transformers").__version__ == "5.19.0", (
            "Use the task-local matching environment"
        )
        c = AutoConfig.from_pretrained(WORK / "base-model", local_files_only=True)
        assert c.model_type == "olmo3" and c.hidden_size == 4096
        assert c.intermediate_size == 11008 and c.num_hidden_layers == 32
        assert c.rms_norm_eps == 1e-6 and c.hidden_act == "silu"
        assert c.sliding_window == 4096 and c.max_position_embeddings == 65536
        assert c.num_attention_heads == c.num_key_value_heads == 32
        assert [i for i, t in enumerate(c.layer_types) if t == "full_attention"] == [
            3,
            7,
            11,
            15,
            19,
            23,
            27,
            31,
        ]
        expected_rope = json.loads(
            (REFERENCE / "checkpoints/native-mxfp4/300461/manifest.json").read_text()
        )["config"]["rope_parameters"]
        assert c.rope_parameters == expected_rope, (c.rope_parameters, expected_rope)
        assert c.rope_parameters["sliding_attention"]["rope_type"] == "default"
        assert c.rope_parameters["full_attention"]["rope_type"] == "yarn"
        with torch.inference_mode():
            rotary = Olmo3RotaryEmbedding(c)
            for layer_type in sorted(set(c.layer_types)):
                cos, sin = rotary(
                    torch.empty(1, dtype=torch.float16),
                    torch.arange(65536)[None],
                    layer_type,
                )
                self.constant("cos_" + layer_type, cos[0, :, :64].numpy())
                self.constant("sin_" + layer_type, sin[0, :, :64].numpy())
        embed = self.constant(
            "embedding", self.source("model.embed_tokens.weight").half().numpy()
        )
        x = self.node("Gather", [embed, "input_ids"], "embedded", axis=0)
        inputs = [
            h.make_tensor_value_info("input_ids", T.INT64, ["batch", "sequence"]),
            h.make_tensor_value_info("position_ids", T.INT64, ["batch", "sequence"]),
            h.make_tensor_value_info("seqlens_k", T.INT32, ["batch"]),
            h.make_tensor_value_info("capacity", T.INT32, []),
            h.make_tensor_value_info("logits_positions", T.INT64, ["logits_sequence"]),
        ]
        outputs = []
        for i in range(self.layers):
            prefix = f"model.layers.{i}"
            q, k, v = [
                self.linear(x, prefix + ".self_attn." + j + "_proj") for j in "qkv"
            ]
            q, k = [
                self.norm(j, prefix + ".self_attn." + t + "_norm")
                for j, t in [(q, "q"), (k, "k")]
            ]
            rotated = []
            for name, value in [("q", q), ("k", k)]:
                f32 = self.node(
                    "Cast", [value], prefix + f".{name}_rope_input_fp32", to=T.FLOAT
                )
                rope = self.node(
                    "RotaryEmbedding",
                    [
                        f32,
                        "position_ids",
                        "cos_" + c.layer_types[i],
                        "sin_" + c.layer_types[i],
                    ],
                    prefix + f".{name}_rope_fp32",
                    domain="com.microsoft",
                    num_heads=32,
                    rotary_embedding_dim=128,
                    interleaved=0,
                )
                rotated.append(
                    self.node("Cast", [rope], prefix + f".{name}_rotated", to=T.FLOAT16)
                )
            q, k = rotated
            for kind in ["key", "value"]:
                inputs.append(
                    h.make_tensor_value_info(
                        f"past_{i}_{kind}",
                        T.FLOAT16,
                        ["batch", 32, "cache_capacity", 128],
                    )
                )
                outputs.append(
                    h.make_tensor_value_info(
                        f"present_{i}_{kind}",
                        T.FLOAT16,
                        ["batch", 32, "cache_capacity", 128],
                    )
                )
            a = self.node(
                "GroupQueryAttention",
                [
                    q,
                    k,
                    v,
                    f"past_{i}_key",
                    f"past_{i}_value",
                    "seqlens_k",
                    "capacity",
                    "",
                    "",
                ],
                [prefix + ".attention", f"present_{i}_key", f"present_{i}_value"],
                domain="com.microsoft",
                num_heads=32,
                kv_num_heads=32,
                scale=128**-0.5,
                do_rotary=0,
                rotary_interleaved=0,
                local_window_size=4096
                if c.layer_types[i] == "sliding_attention"
                else -1,
            )
            o = self.linear(a, prefix + ".self_attn.o_proj")
            o = self.norm(o, prefix + ".post_attention_layernorm")
            x = self.node("Add", [x, o], prefix + ".attention_residual")
            gate, up = [
                self.linear(x, prefix + ".mlp." + j + "_proj") for j in ["gate", "up"]
            ]
            sigmoid = self.node("Sigmoid", [gate], prefix + ".sigmoid")
            activated = self.node("Mul", [gate, sigmoid], prefix + ".silu")
            multiplied = self.node("Mul", [activated, up], prefix + ".gated")
            down = self.linear(multiplied, prefix + ".mlp.down_proj")
            down = self.norm(down, prefix + ".post_feedforward_layernorm")
            x = self.node("Add", [x, down], prefix + ".output")
            print(
                f"Exported Original layer {i}: {self.data.tell() / 2**30:.3f} GiB constants",
                flush=True,
            )
        hidden = self.norm(x, "model.norm")
        selected = self.node(
            "Gather", [hidden, "logits_positions"], "selected_hidden", axis=1
        )
        self.linear(selected, "lm_head", quantized=False)
        self.node("Identity", ["lm_head"], "logits")
        outputs.insert(
            0,
            h.make_tensor_value_info(
                "logits", T.FLOAT16, ["batch", "logits_sequence", c.vocab_size]
            ),
        )
        model = h.make_model(
            h.make_graph(
                self.nodes,
                "Original_OLMo3_RTN_INT4",
                inputs,
                outputs,
                self.initializers,
            ),
            opset_imports=[h.make_opsetid("", 21), h.make_opsetid("com.microsoft", 1)],
            ir_version=10,
        )
        h.set_model_props(
            model,
            {
                "model_revision": REVISION,
                "training_updates": "0",
                "quantization": f"RTN_W{self.bits}A16_group128",
            },
        )
        self.data.close()
        onnx.save(model, self.folder / "original.onnx")
        # ORT's SimplifiedLayerNormalization is a contrib op in the default domain;
        # the stock ONNX checker has no schema. ORT validates it on session load.
        receipt = {
            "model_revision": REVISION,
            "layers": self.layers,
            "training_updates": 0,
            "quantization": f"W{self.bits}A16: FP16 absmax scales; ties-to-even RTN; implicit midpoint; group128",
            "bits": self.bits,
            "rope_parameters": c.rope_parameters,
            "rotary_compute": "FP32 tables and FP32 rotation, then FP16 Q/K boundary",
            "backbone_activations": "FP16",
            "kv": "FP16",
            "lm_head": "FP16",
            "transformers": __import__("transformers").__version__,
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "projections": self.quantization,
            "artifacts": {
                p.name: {"bytes": p.stat().st_size, "sha256": sha(p)}
                for p in self.folder.iterdir()
                if p.is_file() and p.suffix in [".onnx", ".bin"]
            },
        }
        (self.folder / "export.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--bits", type=int, choices=[4, 8], default=4)
    args = parser.parse_args()
    if not 1 <= args.layers <= 32:
        parser.error("layers must be in [1, 32]")
    torch.set_num_threads(8)
    Export(original_folder(args.bits, args.layers), args.layers, args.bits).build()
