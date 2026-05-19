#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Phi-4 (14B) test suite.

Model: microsoft/phi-4-onnx (gpu/gpu-int4-rtn-block-32 variant).
INT4 RTN MatMulNBits block_size=32, 40 layers, 10 KV heads, head_dim=128
(GQA HPG=4). tiktoken cl100k-style BPE tokenizer; <|endoftext|>=100257 = BOS.

Test coverage is provided by `BaseORTTests` (5 tests) + `BaseOGATests`
(4 tests) in conftest.py — see CLAUDE.md "Python Performance Tests".
"""

from conftest import (
    BaseOGATests,
    BaseORTTests,
    ModelSpec,
    REPO_ROOT,
    register_model_fixtures,
)

# ruff: noqa: F811


PHI4 = ModelSpec(
    name="phi4_14b",
    model_dir=REPO_ROOT / "models" / "phi-4-onnx-gpu-int4-rtn-block-32",
    onnx_file="model.onnx",
    data_files=["model.onnx.data"],
    hf_base=(
        "https://huggingface.co/microsoft/phi-4-onnx/resolve/main/"
        "gpu/gpu-int4-rtn-block-32"
    ),
    num_layers=40,
    num_kv_heads=10,
    head_dim=128,
    has_position_ids=True,
    bos_token=100257,
    filler_tokens=[1234, 5678, 9012, 3456, 7890, 12345],
    oga_files=[
        "genai_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ],
)


(
    dynamic_model_path,
    fixed_decode_path,
    fixed_prefill_128_path,
    ep_dynamic_session,
    ep_fixed_decode_session,
    ep_fixed_prefill_128_session,
    oga_default_model,
) = register_model_fixtures(PHI4)


class TestPhi4ORT(BaseORTTests):
    spec = PHI4


class TestPhi4OGA(BaseOGATests):
    spec = PHI4
