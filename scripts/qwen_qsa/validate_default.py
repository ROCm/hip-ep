#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Compare the four-block native graph with a frozen dequantized HF reference."""

import argparse
import json
from pathlib import Path

from verify_handoff import SOURCE, WORK, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--long", action="store_true")
    args = parser.parse_args()
    from default_inference import Default

    import numpy as np
    import onnx
    import torch
    from export_default import source_name
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ForCausalLM,
        Qwen3_5TextRotaryEmbedding,
    )

    torch.set_num_threads(2)
    exported = json.loads((args.folder / "export.json").read_text())
    assert exported["layers"] == 4
    assert sha(args.folder / "default.onnx") == exported["onnx_sha256"]
    assert sha(args.folder / "default.weights.bin") == exported["data_sha256"]
    verified = json.loads((WORK / "handoff-verification.json").read_text())
    snapshot = Path(verified["backbone"]["snapshot"])
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    graph = onnx.load(args.folder / "default.onnx", load_external_data=False).graph
    constants = {x.name: x for x in graph.initializer}
    quantized = {
        source_name(n.input[1].removesuffix(".fresh_rtn_q4")) + ".weight": (
            n.input[1],
            n.input[2],
        )
        for n in graph.node
        if n.op_type == "MatMulNBits"
    }

    def array(name):
        value = constants[name]
        info = {e.key: e.value for e in value.external_data}
        return np.memmap(
            args.folder / info["location"],
            mode="r",
            offset=int(info["offset"]),
            dtype=onnx.helper.tensor_dtype_to_np_dtype(value.data_type),
            shape=tuple(value.dims),
        )

    config = AutoConfig.from_pretrained(snapshot, local_files_only=True).text_config
    config.num_hidden_layers = 4
    config.layer_types = config.layer_types[:4]
    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        reference = Qwen3_5ForCausalLM(config)
    for name, parameter in list(reference.named_parameters()):
        source = (
            name.replace("model.", "model.language_model.", 1)
            if name.startswith("model.")
            else name
        )
        if source in quantized:
            q, s = quantized[source]
            packed, scales = array(q), array(s)
            codes = np.empty((*packed.shape[:-1], packed.shape[-1] * 2), dtype=np.uint8)
            codes[..., ::2], codes[..., 1::2] = packed & 15, packed >> 4
            restored = (codes.astype(np.float32) - 8) * scales.astype(np.float32)[
                ..., None
            ]
            value = torch.from_numpy(restored.reshape(parameter.shape)).to(
                device="cuda", dtype=torch.float16
            )
        else:
            with safe_open(snapshot / index[source], framework="pt") as stream:
                value = stream.get_tensor(source)
            dtype = (
                torch.float32
                if name.endswith((".A_log", ".dt_bias"))
                else torch.float16
            )
            value = value.to(device="cuda", dtype=dtype)
        parent, leaf = name.rsplit(".", 1)
        setattr(
            reference.get_submodule(parent),
            leaf,
            torch.nn.Parameter(value, requires_grad=False),
        )
    reference.model.rotary_emb = Qwen3_5TextRotaryEmbedding(config, device="cuda")
    reference.eval().requires_grad_(False)
    assert all(
        not p.is_meta and p.device.type == "cuda" for p in reference.parameters()
    )
    native = Default(args.folder, capacity=4096 if args.long else 128)
    document = json.loads((SOURCE / "data/qwen-selector-v1/test.json").read_text())[0]
    ids = torch.tensor(
        [document["input_ids"][: 4096 if args.long else 17]],
        device="cuda",
        dtype=torch.int64,
    )
    rows = []
    with torch.inference_mode():
        for count in [1, 7, 17, 513, 4096] if args.long else [1, 7, 17]:
            all_logits = count <= 17
            actual = native.forward(ids[:, :count].contiguous(), all_logits=all_logits)
            expected = reference(
                input_ids=ids[:, :count],
                use_cache=False,
                logits_to_keep=0 if all_logits else 1,
            ).logits
            delta = actual.float() - expected.float()
            relative = (
                (delta.square().mean() / expected.float().square().mean()).sqrt().item()
            )
            maximum = delta.abs().max().item()
            assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
            rows.append({"tokens": count, "relative_rms": relative, "max_abs": maximum})
            print(rows[-1], flush=True)
            assert relative < 0.02, rows[-1]
        boundary = 4095 if args.long else 16
        actual = native.forward(ids[:, :boundary].contiguous())
        actual = native.forward(
            ids[:, boundary : boundary + 1].contiguous(), past=boundary
        )
        prefix = reference(
            input_ids=ids[:, :boundary], use_cache=True, logits_to_keep=1
        )
        expected = reference(
            input_ids=ids[:, boundary : boundary + 1],
            past_key_values=prefix.past_key_values,
            use_cache=True,
        ).logits
        relative = (
            (
                (actual.float() - expected.float()).square().mean()
                / expected.float().square().mean()
            )
            .sqrt()
            .item()
        )
        rows.append({"prefill": boundary, "decode": 1, "relative_rms": relative})
        assert relative < 0.02, rows[-1]
    receipt = {
        "status": "passed",
        "layers": 4,
        "scope": "Numerical architecture diagnostic against frozen dequantized FP16 HF reference; not full-model quality",
        "gate_relative_rms": 0.02,
        "rows": rows,
        "export_onnx_sha256": exported["onnx_sha256"],
        "validator_sha256": sha(Path(__file__)),
        "source_model_revision": verified["revision"],
        "training_updates": 0,
        "selectors_loaded": False,
        "native_gqa_gate": "sigmoid",
        "baseline_established": False,
    }
    (
        WORK
        / ("default-4-reference-long.json" if args.long else "default-4-reference.json")
    ).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
