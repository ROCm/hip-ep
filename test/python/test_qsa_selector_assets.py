#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Public selector migration: frozen measured tensors and fail-closed caching."""

import hashlib
import os
import sys
from pathlib import Path

import onnx
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/qsa"))
from selector_assets import get_selector_asset  # noqa: E402

GOLDEN = {
    "olmo-four": ("qsa/native-gqa/models/Changed", "original.onnx"),
    "qwen-four": ("qwen-qsa/native-gqa/models/Changed", "default.onnx"),
    "qwen-eight": ("qwen-qsa/eight-qsa-v1/models/Eight", "default.onnx"),
}


@pytest.mark.parametrize("key", GOLDEN)
def test_public_tensor_hashes_match_measured_native_export(key):
    relative, graph_name = GOLDEN[key]
    folder = ROOT / "_local" / relative
    if not (folder / graph_name).exists():
        pytest.skip("Measured native export fixture is not mirrored")
    asset = get_selector_asset(key, offline=True)
    graph = onnx.load(folder / graph_name, load_external_data=False)
    expected = asset.provenance["tensor_records"]
    verified = set()
    for tensor in graph.graph.initializer:
        if tensor.name not in expected:
            continue
        assert tensor.data_type == onnx.TensorProto.BFLOAT16
        descriptor = {e.key: e.value for e in tensor.external_data}
        with (folder / descriptor["location"]).open("rb") as stream:
            stream.seek(int(descriptor["offset"]))
            raw = stream.read(int(descriptor["length"]))
        assert hashlib.sha256(raw).hexdigest() == expected[tensor.name]["sha256"]
        verified.add(tensor.name)
    assert verified == set(expected)


@pytest.mark.parametrize("key", GOLDEN)
def test_hf_reexport_is_byte_identical_to_measured_graph_and_selectors(key):
    export_root = os.environ.get("HIP_EP_QSA_HF_EXPORT_ROOT")
    if not export_root:
        pytest.skip("Set HIP_EP_QSA_HF_EXPORT_ROOT to fresh export fixtures")
    relative, graph_name = GOLDEN[key]
    golden = ROOT / "_local" / relative
    fresh = Path(export_root) / key
    if key != "qwen-eight":
        fresh /= "Changed"
    for name in [graph_name, "qsa.selectors.bin"]:
        assert (fresh / name).read_bytes() == (golden / name).read_bytes(), (key, name)


def test_offline_missing_asset_does_not_download(tmp_path, monkeypatch):
    def reject_download(*args, **kwargs):
        pytest.fail("Offline lookup attempted a network request")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", reject_download)
    with pytest.raises(FileNotFoundError, match="Missing cached selector asset"):
        get_selector_asset("qwen-four", cache_root=tmp_path, offline=True)


def test_corrupt_metadata_is_rejected_before_use(tmp_path):
    folder = tmp_path / "qwen-eight"
    folder.mkdir()
    (folder / "selector_config.json").write_text("{}")
    with pytest.raises(ValueError, match="Selector checksum mismatch"):
        get_selector_asset("qwen-eight", cache_root=tmp_path, offline=True)


@pytest.mark.parametrize(
    "command,model,key",
    [
        (["scripts/qsa/export_native_gqa.py"], "olmo", "olmo-four"),
        (["scripts/qwen_qsa/export_native_gqa.py"], "qwen", "qwen-four"),
        (["scripts/qwen_qsa/export_eight.py"], "qwen", "qwen-eight"),
        (
            ["scripts/qwen_qsa/bfcl_worker.py", "--variant", "Eight"],
            "qwen",
            "qwen-eight",
        ),
    ],
)
def test_stage_selects_public_asset_without_private_handoff(
    command, model, key, monkeypatch
):
    import run_stage

    monkeypatch.setattr(run_stage, "get_selector_asset", lambda selected: selected)
    assert run_stage.selector_for_command(command) == (model, key)
