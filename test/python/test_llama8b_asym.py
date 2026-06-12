#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Llama-3.1-8B AWQ INT4 g128 **asymmetric** test suite.

Model: amd/Llama-3.1-8B-awq-g128-int4-asym-fp16-onnx-dml (gated AMD repo).
MatMulNBits block_size=128 + `zero_points` (packed uint8 nibbles) — exercises
the asym runtime path through `zp_unpack_cache` (CLAUDE.md "Asym
MatMulNBits zero_points unpack cache" gotcha). This is the **base** Llama-3.1
(not the Instruct fine-tune like the sym repo) — different greedy generations
expected; see CLAUDE.md "Sym vs asym 8B model identity" gotcha.

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


LLAMA8B_ASYM = ModelSpec(
    name="llama8b_asym",
    model_dir=REPO_ROOT / "models" / "Llama-3.1-8B-awq-g128-int4-asym-fp16-onnx-dml",
    onnx_file="model.onnx",
    data_files=["model.onnx.data"],
    hf_repo="amd/Llama-3.1-8B-awq-g128-int4-asym-fp16-onnx-dml",
    num_layers=32,
    num_kv_heads=8,
    head_dim=128,
    has_position_ids=True,
    bos_token=128000,
    filler_tokens=[9906, 11, 1268, 527, 499, 30],
    oga_files=[
        "genai_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
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
) = register_model_fixtures(LLAMA8B_ASYM)


class TestLlama8BAsymORT(BaseORTTests):
    spec = LLAMA8B_ASYM


class TestLlama8BAsymOGA(BaseOGATests):
    spec = LLAMA8B_ASYM
