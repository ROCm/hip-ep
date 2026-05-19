#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Mistral-7B-Instruct-v0.3 test suite.

Model: EmbeddedLLM/mistral-7b-instruct-v0.3-int4-onnx-directml.
32 layers, 8 KV heads, head_dim=128. SentencePiece tokenizer (BOS=<s>=1).

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


MISTRAL7B = ModelSpec(
    name="mistral7b_v3",
    model_dir=REPO_ROOT / "models" / "mistral-7b-instruct-v0.3-int4-onnx-directml",
    onnx_file="model.onnx",
    data_files=["model.onnx.data"],
    hf_base=(
        "https://huggingface.co/EmbeddedLLM/"
        "mistral-7b-instruct-v0.3-int4-onnx-directml/resolve/main"
    ),
    num_layers=32,
    num_kv_heads=8,
    head_dim=128,
    has_position_ids=True,
    bos_token=1,
    filler_tokens=[1234, 5678, 9012, 3456, 7890, 12345],
    oga_files=[
        "genai_config.json",
        "tokenizer.json",
        "tokenizer.model",
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
) = register_model_fixtures(MISTRAL7B)


class TestMistral7BV3ORT(BaseORTTests):
    spec = MISTRAL7B


class TestMistral7BV3OGA(BaseOGATests):
    spec = MISTRAL7B
