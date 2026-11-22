#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Record CPU-only Qwen baseline constraints and test the executed native gate."""

import collections
import hashlib
import inspect
import json

import onnx
import psutil
import torch
import transformers
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention
from verify_handoff import ROOT, WORK


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.inference_mode()
def gate_control():
    """A single key makes attention output independent of query/key scores."""
    config = Qwen3_5TextConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        layer_types=["full_attention"],
        output_gate_type="swish",
    )
    config._attn_implementation = "eager"
    attention = Qwen3_5Attention(config, 0).eval()
    attention.q_proj.weight.zero_()
    attention.q_proj.weight[4:8, 0] = 1
    attention.q_proj.weight[12:16, 0] = 1
    attention.v_proj.weight.zero_()
    attention.v_proj.weight[:4, :4] = torch.eye(4)
    attention.o_proj.weight.copy_(torch.eye(8))
    hidden = torch.ones(1, 1, 8)
    hidden[0, 0, 0] = 2
    output = attention(
        hidden,
        position_embeddings=(torch.ones(1, 1, 2), torch.zeros(1, 1, 2)),
        attention_mask=None,
    )[0]
    value = hidden[:, :, :4].repeat(1, 1, 2)
    sigmoid = value * torch.sigmoid(torch.tensor(2.0))
    swish = sigmoid * 2
    error = float((output - sigmoid).abs().max())
    alternative_error = float((output - swish).abs().max())
    assert error == 0 and alternative_error > 1
    return {
        "device": "cpu",
        "config_output_gate_type": config.output_gate_type,
        "executed_gate": "sigmoid",
        "sigmoid_max_abs_error": error,
        "swish_max_abs_error": alternative_error,
        "scope": "Synthetic native-class control, not a model accuracy result",
        "native_forward_sha256": hashlib.sha256(
            inspect.getsource(Qwen3_5Attention.forward).encode()
        ).hexdigest(),
    }


def main():
    handoff = json.loads((WORK / "handoff-verification.json").read_text())
    assert handoff["backbone"]["weight_bytes_verified"]
    template = ROOT.parent / "hip-ep-turboquant/models/qwen-original/text.onnx"
    expected = "44b712df872250c1c1614c3aa717788a103f3a814e0cfd24b22196d3b99509cd"
    # The pinned template is inspected only. It is not the new backbone.
    actual = sha(template)
    assert actual == expected, actual
    graph = onnx.load(template, load_external_data=False).graph
    operators = collections.Counter(n.op_type for n in graph.node)
    assert operators["LinearAttention"] == operators["CausalConvWithState"] == 48
    assert operators["GroupQueryAttention"] == 16
    states = [x for x in graph.input if x.name.endswith(".recurrent_state")]
    assert len(states) == 48
    assert all(x.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16 for x in states)
    receipt = {
        "status": "passed",
        "scope": "CPU source/graph reconnaissance and synthetic gate control",
        "transformers": transformers.__version__,
        "torch": str(torch.__version__),
        "gpu_allocations": 0,
        "baseline_established": False,
        "training_updates": 0,
        "gate": gate_control(),
        "ram_total_bytes": psutil.virtual_memory().total,
        "ram_available_bytes": psutil.virtual_memory().available,
        "bf16_backbone_including_vision_mtp_bytes": handoff["backbone"]["tensor_bytes"],
        "template_sha256": actual,
        "template_operators": dict(operators),
        "template_recurrent_io_dtype": "FP16",
        "required_recurrent_io_dtype": "FP32",
        "constraints": [
            "Native GQA currently supports FP16/FP32, not BF16.",
            "Full BF16/FP16 snapshot requires additional cache, workspace and host memory.",
            "Template learned constants cannot substitute for the verified official backbone.",
            "LinearAttention runtime shares one dtype across query, key, value and state.",
            "An FP32 LinearAttention region with explicit casts can preserve FP32 state.",
            "Preserve executed native sigmoid gating despite the extra swish config field.",
            "Any fresh RTN representation needs its own default numerical/accuracy controls.",
        ],
        "source_sha256": {
            name: sha(ROOT / name)
            for name in [
                "lib/Runtime/real/gqa.cpp",
                "lib/Runtime/real/linear_attention.cpp",
                "lib/Conversion/HipToLLVM/LinearAttentionLowering.cpp",
            ]
        },
    }
    output = WORK / "native-support.json"
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
