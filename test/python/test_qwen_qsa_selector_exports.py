#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Verify both pinned Qwen selector exports without running a GPU model."""

import hashlib
import json
import os
from pathlib import Path

import onnx
import pytest

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = Path(os.environ.get("HIP_EP_QSA_ARTIFACT_ROOT", ROOT))


@pytest.mark.parametrize(
    "relative,layers,selector_sha256",
    [
        (
            "native-gqa/models/Changed",
            [27, 31, 35, 39],
            "d7729599f421d80ace536f84145bafa9a6e1eaad22dde18ce42392014a58b5e4",
        ),
        (
            "eight-qsa-v1/models/Eight",
            [3, 7, 23, 27, 31, 35, 39, 63],
            "8915ef00df9ad755808591862b473ee0e265cfd405ce1fc787bf948aea56eeeb",
        ),
    ],
    ids=["four", "eight"],
)
def test_qwen_selector_exports_preserve_backbone_and_frozen_tensors(
    relative, layers, selector_sha256
):
    work = ARTIFACT_ROOT / "_local/qwen-qsa"
    source = work / "native-gqa/models/Original"
    folder = work / relative
    if not (folder / "default.onnx").exists():
        pytest.skip("Pinned local Qwen export is not mirrored")
    receipt = json.loads((folder / "export.json").read_text())
    baseline = json.loads((source / "export.json").read_text())
    assert receipt["selector_sha256"] == selector_sha256
    assert receipt["baseline_onnx_sha256"] == baseline["onnx_sha256"]
    assert receipt["data_sha256"] == baseline["data_sha256"]
    assert (
        hashlib.sha256((folder / "default.onnx").read_bytes()).hexdigest()
        == receipt["onnx_sha256"]
    )
    assert (
        hashlib.sha256((folder / "qsa.selectors.bin").read_bytes()).hexdigest()
        == receipt["selector_data_sha256"]
    )
    before = onnx.load(source / "default.onnx", load_external_data=False).graph
    after = onnx.load(folder / "default.onnx", load_external_data=False).graph
    assert len(before.node) == len(after.node)
    replaced = []
    for old, new in zip(before.node, after.node, strict=True):
        if new.op_type != "QwenQsaGroupQueryAttention":
            assert old.SerializeToString() == new.SerializeToString()
            continue
        layer = int(new.input[3].split(".")[1])
        replaced.append(layer)
        assert new.domain == "com.amd" and len(new.input) == 15
        assert old.op_type == "GroupQueryAttention"
        assert list(new.input[:7]) == list(old.input[:7])
        assert list(new.output) == list(old.output)
        assert new.input[7] == f"/model/layers.{layer}/input_layernorm/output_0"
        assert new.input[-1] == "position_ids"
    assert replaced == layers
    assert sum(n.op_type == "LinearAttention" for n in after.node) == 48
    original = {t.name: t.SerializeToString() for t in before.initializer}
    updated = {t.name: t for t in after.initializer}
    assert all(
        updated[name].SerializeToString() == raw for name, raw in original.items()
    )
    selectors = [t for name, t in updated.items() if name not in original]
    assert len(selectors) == 4 * len(layers)
    assert {t.name for t in selectors} == set(receipt["selector_tensor_sha256"])
    for tensor in selectors:
        assert tensor.data_type == onnx.TensorProto.BFLOAT16
        external = {e.key: e.value for e in tensor.external_data}
        assert external["location"] == "qsa.selectors.bin"
        with (folder / external["location"]).open("rb") as stream:
            stream.seek(int(external["offset"]))
            raw = stream.read(int(external["length"]))
        assert (
            hashlib.sha256(raw).hexdigest()
            == receipt["selector_tensor_sha256"][tensor.name]
        )
