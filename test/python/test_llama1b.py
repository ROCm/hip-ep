#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Llama-3.2-1B-Instruct test suite.

Model: q4f16, 16 layers, 8 KV heads, head_dim=64. No `position_ids` input.
The HF repo lacks `genai_config.json` so we ship one in
`_GENAI_CONFIG_TEMPLATE`; tokenizer files are downloaded from the repo root.

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

# ruff: noqa: F811  # pytest fixtures shadow function names

# 1B repo has no genai_config.json on HF — we generate this template on first
# OGA test. Kept inline rather than fetched.
_GENAI_CONFIG_TEMPLATE = {
    "model": {
        "bos_token_id": 128000,
        "context_length": 131072,
        "decoder": {
            "session_options": {
                "log_id": "onnxruntime-genai",
                "provider_options": [],
            },
            "filename": "model_q4f16.onnx",
            "head_size": 64,
            "hidden_size": 2048,
            "inputs": {
                "input_ids": "input_ids",
                "attention_mask": "attention_mask",
                "past_key_names": "past_key_values.%d.key",
                "past_value_names": "past_key_values.%d.value",
            },
            "outputs": {
                "logits": "logits",
                "present_key_names": "present.%d.key",
                "present_value_names": "present.%d.value",
            },
            "num_attention_heads": 32,
            "num_hidden_layers": 16,
            "num_key_value_heads": 8,
        },
        "eos_token_id": [128001, 128008, 128009],
        "pad_token_id": 128001,
        "type": "llama",
        "vocab_size": 128256,
    },
    "search": {
        "chunk_size": 1024,
        "diversity_penalty": 0.0,
        "do_sample": True,
        "early_stopping": True,
        "length_penalty": 1.0,
        "max_length": 131072,
        "min_length": 0,
        "no_repeat_ngram_size": 0,
        "num_beams": 1,
        "num_return_sequences": 1,
        "past_present_share_buffer": True,
        "repetition_penalty": 1.0,
        "temperature": 0.6,
        "top_k": 1,
        "top_p": 0.9,
    },
}


LLAMA1B = ModelSpec(
    name="llama1b",
    model_dir=REPO_ROOT / "models" / "Llama-3.2-1B-Instruct",
    onnx_file="model_q4f16.onnx",
    data_files=["model_q4f16.onnx_data"],
    hf_base=(
        "https://huggingface.co/onnx-community/Llama-3.2-1B-Instruct-ONNX/"
        "resolve/main/onnx"
    ),
    num_layers=16,
    num_kv_heads=8,
    head_dim=64,
    has_position_ids=False,
    bos_token=128000,
    filler_tokens=[9906, 11, 1268, 527, 499, 30],
    oga_files=[
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ],
    genai_config_template=_GENAI_CONFIG_TEMPLATE,
    hf_root_for_tokenizer=(
        "https://huggingface.co/onnx-community/Llama-3.2-1B-Instruct-ONNX/resolve/main"
    ),
    # Run the output-allocator (2-arg ABI) e2e accuracy test on this fast model;
    # allocator mode is model-agnostic EP plumbing so 1B is representative.
    output_allocator_e2e=True,
)


(
    dynamic_model_path,
    fixed_decode_path,
    fixed_prefill_128_path,
    ep_dynamic_session,
    ep_fixed_decode_session,
    ep_fixed_prefill_128_session,
    oga_default_model,
) = register_model_fixtures(LLAMA1B)


class TestLlama1BORT(BaseORTTests):
    spec = LLAMA1B


class TestLlama1BOGA(BaseOGATests):
    spec = LLAMA1B
