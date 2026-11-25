#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Inference export guards for the pinned OLMo QSA experiment (GPU-free)."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/qsa"))
from export_original import quantize  # noqa: E402
from prepare_baseline import SELECTOR, SELECTOR_SHA, original_folder  # noqa: E402


@pytest.mark.parametrize("bits", [4, 8])
def test_packed_weights_round_trip(bits):
    values = torch.linspace(-1, 1, 4 * 128).reshape(4, 128)
    packed, scales = quantize(values, bits=bits)
    if bits == 4:
        # Independently unpack each byte in low-then-high nibble order.
        codes = np.stack([packed & 15, packed >> 4], axis=-1).reshape(4, 1, 128)
    else:
        codes = packed
    decoded = (codes.astype(np.float32) - 2 ** (bits - 1)) * scales.astype(np.float32)[
        ..., None
    ]
    assert (
        np.max(np.abs(decoded.reshape(4, 128) - values.numpy()))
        <= scales.max() / 2 + 1e-6
    )
    assert packed.dtype == np.uint8 and scales.dtype == np.float16


def test_selector_is_the_frozen_final_bf16_export():
    if not SELECTOR.exists():
        pytest.skip("Local frozen selector fixture is not mirrored")
    assert hashlib.sha256(SELECTOR.read_bytes()).hexdigest() == SELECTOR_SHA
    export = torch.load(SELECTOR, map_location="cpu", weights_only=True)
    assert export["architecture"]["layers"] == [7, 11, 15, 23]
    state = export["selector"]
    assert state and all(t.dtype == torch.bfloat16 for t in state.values())


def test_export_retains_projections_and_per_layer_rope():
    folder = original_folder()
    if not (folder / "export.json").exists():
        pytest.skip("Faithful export has not completed")
    receipt = json.loads((folder / "export.json").read_text())
    assert receipt["transformers"] == "5.19.0"
    assert receipt["rope_parameters"]["sliding_attention"]["rope_type"] == "default"
    assert receipt["rope_parameters"]["full_attention"]["rope_type"] == "yarn"
    model = onnx.load(folder / "original.onnx", load_external_data=False)
    projections = [n for n in model.graph.node if n.op_type == "MatMulNBits"]
    assert len(projections) == 224 and len(receipt["projections"]) == 224
    attention = [n for n in model.graph.node if n.op_type == "GroupQueryAttention"]
    assert len(attention) == 32
    rotary = [n for n in model.graph.node if n.op_type == "RotaryEmbedding"]
    assert len(rotary) == 64
    for index, node in enumerate(rotary):
        i = index // 2
        kind = "full_attention" if i % 4 == 3 else "sliding_attention"
        assert node.input[2:4] == ["cos_" + kind, "sin_" + kind]


def test_native_qsa_graph_retains_backbone_and_exact_bf16_parameters():
    native = ROOT / "_local/qsa/native-gqa/models/Changed"
    if not (native / "original.onnx").exists():
        pytest.skip("Native QSA export is not mirrored")
    model = onnx.load(native / "original.onnx", load_external_data=False)
    source = onnx.load(original_folder() / "original.onnx", load_external_data=False)
    assert len(model.graph.node) == len(source.graph.node)
    sparse = [n for n in model.graph.node if n.op_type == "QsaGroupQueryAttention"]
    assert len(sparse) == 4
    assert all(
        n.domain == "com.amd" and len(n.input) == 13 and len(n.output) == 3
        for n in sparse
    )
    assert {int(n.input[0].split(".")[2]) for n in sparse} == {7, 11, 15, 23}
    for node, before in zip(model.graph.node, source.graph.node, strict=True):
        if node.op_type != "QsaGroupQueryAttention":
            assert node.SerializeToString() == before.SerializeToString()
    frozen = torch.load(SELECTOR, map_location="cpu", weights_only=True)["selector"]
    params = [x for x in model.graph.initializer if x.name.startswith("selectors.")]
    assert len(params) == 12
    for param in params:
        assert param.data_type == onnx.TensorProto.BFLOAT16
        data = {e.key: e.value for e in param.external_data}
        with (native / data["location"]).open("rb") as file:
            file.seek(int(data["offset"]))
            raw = file.read(int(data["length"]))
        assert (
            raw == frozen[param.name].contiguous().view(torch.uint8).numpy().tobytes()
        )


def test_native_qwen_changes_only_four_nodes_and_frozen_selectors():
    folder = ROOT / "_local/qwen-qsa/native-gqa/models"
    if not (folder / "Changed/default.onnx").exists():
        pytest.skip("Native Qwen exports are not mirrored")
    variants = {
        name: onnx.load(folder / name / "default.onnx", load_external_data=False)
        for name in ["Original", "Changed"]
    }
    original, changed = [variants[name].graph for name in ["Original", "Changed"]]
    assert len(original.node) == len(changed.node)
    replacements = []
    for before, after in zip(original.node, changed.node, strict=True):
        if after.op_type == "QwenQsaGroupQueryAttention":
            layer = int(after.input[3].split(".")[1])
            replacements.append(layer)
            assert after.domain == "com.amd" and len(after.input) == 15
            assert list(before.output) == list(after.output)
            assert list(before.input[:7]) == list(after.input[:7])
            assert after.input[7] == f"/model/layers.{layer}/input_layernorm/output_0"
            assert after.input[-1] == "position_ids"
        else:
            assert before.SerializeToString() == after.SerializeToString()
    assert replacements == [27, 31, 35, 39]
    assert sum(node.op_type == "LinearAttention" for node in changed.node) == 48
    before = {item.name: item.SerializeToString() for item in original.initializer}
    after = {item.name: item for item in changed.initializer}
    assert all(after[name].SerializeToString() == raw for name, raw in before.items())
    selectors = [item for name, item in after.items() if name not in before]
    assert len(selectors) == 16
    receipt = json.loads((folder / "Changed/export.json").read_text())
    for tensor in selectors:
        assert tensor.data_type == onnx.TensorProto.BFLOAT16
        descriptor = {item.key: item.value for item in tensor.external_data}
        with (folder / "Changed" / descriptor["location"]).open("rb") as stream:
            stream.seek(int(descriptor["offset"]))
            raw = stream.read(int(descriptor["length"]))
        assert (
            hashlib.sha256(raw).hexdigest()
            == receipt["selector_tensor_sha256"][tensor.name]
        )
