#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""gpt-oss-20b test suite.

Model: onnxruntime/gpt-oss-20b-onnx (webgpu-int4-rtn-block-32 variant).
24 layers (alternating sliding/full attention, sliding_window=128 baked into
the graph), 64 attention heads, 8 KV heads (HPG=8), head_dim=64, hidden=2880,
vocab=201088. MoE backbone (32 experts, 4 active per token, MXFP4 weights).
attention_bias=true, YaRN rope. **No `position_ids` input** — rope is driven
internally by the graph.

The HF repo splits weights across 7 external-data blobs.

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


GPTOSS20B = ModelSpec(
    name="gptoss20b",
    model_dir=REPO_ROOT / "models" / "gpt-oss-20b-int4-rtn-block-32",
    onnx_file="model_q4f16.onnx",
    # 7 external-data blobs — HF converter split ~12.5 GB to stay under
    # storage backend per-file caps.
    data_files=[
        "model_q4f16.onnx_data",
        "model_q4f16.onnx_data_1",
        "model_q4f16.onnx_data_2",
        "model_q4f16.onnx_data_3",
        "model_q4f16.onnx_data_4",
        "model_q4f16.onnx_data_5",
        "model_q4f16.onnx_data_6",
    ],
    hf_base=(
        "https://huggingface.co/onnxruntime/gpt-oss-20b-onnx/resolve/main/"
        "webgpu/webgpu-int4-rtn-block-32"
    ),
    num_layers=24,
    num_kv_heads=8,
    head_dim=64,
    has_position_ids=False,
    bos_token=199998,
    # Avoid the special-token cluster (>=199998). Tests compare CPU-vs-EP
    # logits — decoded text is irrelevant.
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
) = register_model_fixtures(GPTOSS20B)


class TestGptOss20BORT(BaseORTTests):
    spec = GPTOSS20B


class TestGptOss20BOGA(BaseOGATests):
    spec = GPTOSS20B
