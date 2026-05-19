#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""DeepSeek-R1-Distill-Llama-70B test suite.

Model: amd/DeepSeek-R1-Distill-Llama-70B-dml-int4-awq-block-128 (gated AMD
repo). AWQ INT4 MatMulNBits block_size=128, 80 layers, 8 KV heads (HPG=8),
head_dim=128. Llama-3 BPE tokenizer (BOS=128000). Auth via `hf auth login`.

`normalize_drop_inputs_embeds` defends against the dangling `inputs_embeds`
graph input quirk present in sibling AMD repos.

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


DEEPSEEK_R1_70B = ModelSpec(
    name="deepseek_r1_70b",
    model_dir=(
        REPO_ROOT / "models" / "DeepSeek-R1-Distill-Llama-70B-dml-int4-awq-block-128"
    ),
    onnx_file="model.onnx",
    data_files=["model.onnx.data"],
    hf_repo="amd/DeepSeek-R1-Distill-Llama-70B-dml-int4-awq-block-128",
    num_layers=80,
    num_kv_heads=8,
    head_dim=128,
    has_position_ids=True,
    bos_token=128000,
    filler_tokens=[1234, 5678, 9012, 3456, 7890, 12345],
    oga_files=[
        "genai_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ],
    normalize_onnx_hook=normalize_drop_inputs_embeds,
    # 70B = ~80 GB constants. Module-scope session reuse would keep up to
    # three sessions alive across tests; combined with the EP's
    # SHARED_CONSTANTS pin that exceeds the 96 GB Strix Halo budget. Fall
    # back to per-test session creation — matches the pre-2026-05 baseline.
    reuse_ep_session=False,
    reuse_oga_default_model=False,
)


(
    dynamic_model_path,
    fixed_decode_path,
    fixed_prefill_128_path,
    ep_dynamic_session,
    ep_fixed_decode_session,
    ep_fixed_prefill_128_session,
    oga_default_model,
) = register_model_fixtures(DEEPSEEK_R1_70B)


class TestDeepSeekR170BORT(BaseORTTests):
    spec = DEEPSEEK_R1_70B


class TestDeepSeekR170BOGA(BaseOGATests):
    spec = DEEPSEEK_R1_70B
