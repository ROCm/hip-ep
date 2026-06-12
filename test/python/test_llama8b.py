#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Llama-3.1-8B-Instruct AWQ INT4 g128 symmetric test suite.

Model: amd/Llama-3.1-8B-Instruct-awq-g128-int4-onnx-directml
(MatMulNBits block_size=128, **symmetric** — no zero_points,
32 layers, 8 KV heads, head_dim=128).

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


LLAMA8B = ModelSpec(
    name="llama8b",
    model_dir=REPO_ROOT / "models" / "Llama-3.1-8B-Instruct-awq-g128-int4",
    onnx_file="model.onnx",
    data_files=["model.onnx.data"],
    hf_base=(
        "https://huggingface.co/amd/"
        "Llama-3.1-8B-Instruct-awq-g128-int4-onnx-directml/resolve/main"
    ),
    num_layers=32,
    num_kv_heads=8,
    head_dim=128,
    has_position_ids=True,
    bos_token=128000,
    filler_tokens=[9906, 11, 1268, 527, 499, 30],
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
) = register_model_fixtures(LLAMA8B)


class TestLlama8BORT(BaseORTTests):
    spec = LLAMA8B


class TestLlama8BOGA(BaseOGATests):
    spec = LLAMA8B
