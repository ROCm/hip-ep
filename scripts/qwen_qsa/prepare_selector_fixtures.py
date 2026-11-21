#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Freeze CPU feature/index oracles without adding QSA to native model inference."""

import ast
import json
from pathlib import Path

import numpy as np
import onnx
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from selector_reference import features, select, token_support
from verify_handoff import PAYLOAD_SHA, SOURCE, WORK, sha


def main():
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    output = WORK / "selector-fixtures-v1"
    output.mkdir(exist_ok=False)
    payload = (
        SOURCE / "checkpoints/hip-ep-qwen/303042-step-0128/selector-bf16.safetensors"
    )
    assert sha(payload) == PAYLOAD_SHA
    source_path = SOURCE / "scripts/qwen_qsa.py"
    identity = json.loads(
        (SOURCE / "docs/reproducibility/qwen-trained-performance.json").read_text()
    )
    assert sha(source_path) == identity["runtime"]["scripts/qwen_qsa.py"]["measured"]
    # Execute only the three pure reference definitions. No source imports,
    # extension building, GPU access or source-project bytecode writes.
    tree = ast.parse(source_path.read_text())
    definitions = [
        n
        for n in tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef))
        and n.name in {"Norm", "rotary", "QwenSelector"}
    ]
    assert len(definitions) == 3
    namespace = {"torch": torch, "nn": torch.nn}
    exec(
        compile(
            ast.Module(body=definitions, type_ignores=[]), str(source_path), "exec"
        ),
        namespace,
    )
    exported = json.loads((WORK / "default-rtn-w4a16-64/export.json").read_text())
    model_path = WORK / "default-rtn-w4a16-64/default.onnx"
    assert sha(model_path) == exported["onnx_sha256"]
    graph = onnx.load(model_path, load_external_data=False).graph
    initializers = {x.name: x for x in graph.initializer}
    length = 2064
    cos_sin = []
    for name in ["model.rotary_emb.cos_cache", "model.rotary_emb.sin_cache"]:
        value = initializers[name]
        assert value.data_type == onnx.TensorProto.FLOAT and list(value.dims) == [
            262144,
            32,
        ]
        descriptor = {x.key: x.value for x in value.external_data}
        mapped = np.memmap(
            model_path.parent / descriptor["location"],
            dtype=np.float32,
            mode="r",
            offset=int(descriptor["offset"]),
            shape=tuple(value.dims),
        )
        cos_sin.append(torch.from_numpy(mapped[:length].copy())[None].repeat(1, 1, 2))
    cos, sin = cos_sin
    generator = torch.Generator(device="cpu").manual_seed(20261126)
    hidden = torch.randn(1, length, 5120, generator=generator).to(torch.float16)
    positions = [
        0,
        1,
        2,
        3,
        4,
        7,
        8,
        2047,
        2048,
        2049,
        2050,
        2051,
        2052,
        2053,
        2054,
        2055,
        2056,
        2059,
        2063,
    ]
    tensors = {
        "hidden_fp16": hidden,
        "cos_fp32": cos,
        "sin_fp32": sin,
        "query_positions": torch.tensor(positions, dtype=torch.int32),
    }
    checks = []
    with (
        torch.inference_mode(),
        safe_open(payload, framework="pt", device="cpu") as weights_file,
    ):
        for layer in [27, 31, 35, 39]:
            prefix = f"selectors.{layer}."
            weights = {
                key.removeprefix(prefix): weights_file.get_tensor(key)
                for key in weights_file.keys()
                if key.startswith(prefix)
            }
            assert len(weights) == 4 and all(
                x.dtype == torch.bfloat16 for x in weights.values()
            )
            actual = features(hidden, weights, cos, sin)
            canonical = (
                namespace["QwenSelector"]()
                .to(dtype=torch.bfloat16)
                .eval()
                .requires_grad_(False)
            )
            canonical.load_state_dict(weights, strict=True)
            expected_query, expected_key = canonical.project(
                hidden.to(torch.bfloat16), cos, sin
            )
            expected_pool = canonical.pool(expected_key, cos, sin)
            assert torch.equal(actual["query"], expected_query)
            assert torch.equal(actual["token_key"], expected_key)
            assert torch.equal(actual["pooled_key"], expected_pool)
            assert torch.equal(
                actual["projected"], canonical.projection(hidden.to(torch.bfloat16))
            )
            indices = torch.stack(
                [
                    select(
                        actual["query"][0, position], actual["pooled_key"][0], position
                    )
                    for position in positions
                ]
            )
            counts = [
                len(token_support(row, position))
                for row, position in zip(indices, positions)
            ]
            assert counts[positions.index(2051)] == 2052
            assert counts[positions.index(2052)] == 2049
            assert counts[positions.index(2055)] == 2052
            for key, value in actual.items():
                tensors[f"layer{layer}.{key}"] = value.contiguous()
            tensors[f"layer{layer}.selected_groups"] = indices
            checks.append(
                {
                    "layer": layer,
                    "canonical_features_bit_exact": True,
                    "support_counts": counts,
                    "nonzero_norm_counts": {
                        key: int((value != 0).sum())
                        for key, value in weights.items()
                        if "norm" in key
                    },
                }
            )
            print("CPU canonical selector verified", layer, flush=True)
    tied = select(
        torch.zeros(4, 128, dtype=torch.bfloat16),
        torch.zeros(516, 128, dtype=torch.bfloat16),
        2052,
    )
    assert tied.tolist() == list(range(512)) + [513]
    tensors["threshold_tie_selected_groups"] = tied
    path = output / "fixtures.safetensors"
    save_file(tensors, path)
    assert sha(payload) == PAYLOAD_SHA and not torch.cuda.is_initialized()
    receipt = {
        "status": "passed",
        "scope": "CPU fixtures match canonical frozen source features; no native QSA qualification or GPU inference",
        "selector_sha256": PAYLOAD_SHA,
        "native_default_graph_sha256": exported["onnx_sha256"],
        "canonical_source_sha256": sha(source_path),
        "fixtures_sha256": sha(path),
        "fixture_script_sha256": sha(Path(__file__)),
        "reference_script_sha256": sha(
            Path(__file__).with_name("selector_reference.py")
        ),
        "input_hidden_dtype": "FP16 cast to BF16 for frozen projection",
        "rope_dtype": "Actual native FP32 cos/sin",
        "seed": 20261126,
        "tokens": length,
        "positions": positions,
        "checks": checks,
        "threshold_ties_prefer_earlier_indices": True,
        "training_updates": 0,
        "gpu_allocations": 0,
        "native_qsa_integrated": False,
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps({"status": "passed", "sha256": receipt["fixtures_sha256"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
