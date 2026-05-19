#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Qwen2.5-Coder-14B-Instruct test suite.

Model: amd/Qwen2.5-Coder-14B-instruct-rtn-128gs-fp16-onnx-gpu (gated AMD repo).
Same architecture as the base Qwen2.5-14B-Instruct (qwen2 / 48 layers /
8 KV heads / HPG=5 / head_dim=128), different post-training. Same
`Upload Qwen2.5 ONNX files/` subdir layout, same dangling-`inputs_embeds`
graph-input quirk that `normalize_drop_inputs_embeds` strips.

Test coverage is provided by `BaseORTTests` (5 tests) + `BaseOGATests`
(4 tests) in conftest.py — see CLAUDE.md "Python Performance Tests".
"""

from conftest import (
    BaseOGATests,
    BaseORTTests,
    ModelSpec,
    REPO_ROOT,
    normalize_drop_inputs_embeds,
    register_model_fixtures,
)

# ruff: noqa: F811


QWEN25_CODER_14B = ModelSpec(
    name="qwen2_5_coder_14b",
    model_dir=(
        REPO_ROOT / "models" / "Qwen2.5-Coder-14B-instruct-rtn-128gs-fp16-onnx-gpu"
    ),
    onnx_file="text.onnx",
    data_files=["text.onnx.data"],
    hf_repo="amd/Qwen2.5-Coder-14B-instruct-rtn-128gs-fp16-onnx-gpu",
    hf_subdir="Upload Qwen2.5 ONNX files",
    num_layers=48,
    num_kv_heads=8,
    head_dim=128,
    has_position_ids=True,
    bos_token=151643,
    filler_tokens=[1234, 5678, 9012, 3456, 7890, 12345],
    oga_files=[
        "added_tokens.json",
        "chat_template.jinja",
        "config.json",
        "genai_config.json",
        "generation_config.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ],
    normalize_onnx_hook=normalize_drop_inputs_embeds,
)


(
    dynamic_model_path,
    fixed_decode_path,
    fixed_prefill_128_path,
    ep_dynamic_session,
    ep_fixed_decode_session,
    ep_fixed_prefill_128_session,
    oga_default_model,
) = register_model_fixtures(QWEN25_CODER_14B)


class TestQwen2_5_Coder_14BORT(BaseORTTests):
    spec = QWEN25_CODER_14B


class TestQwen2_5_Coder_14BOGA(BaseOGATests):
    spec = QWEN25_CODER_14B
