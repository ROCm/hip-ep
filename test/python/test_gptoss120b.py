#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""gpt-oss-120b test suite.

Model: amd/gpt-oss-120b-w-uint4-pergroup-asym-awq-onnx-fp16 (gated AMD repo).
AWQ INT4 pergroup-asymmetric MoE (128 experts, 4 active per token,
group_size=32) + fp16 attention. 36 layers (alternating sliding/full,
sliding_window=128), 64 attention heads, 8 KV heads (HPG=8), head_dim=64,
hidden=2880, vocab=201088. Auth via `hf auth login`.

Two environment limits constrain coverage:

1. ORT CPU EP's `QMoE` op only supports per-channel scales — this model
   uses per-group (90-element last-dim). `cpu_skip_predicate` maps the
   resulting error (and Python heap bad_alloc after multiple 64 GB loads)
   to `pytest.skip`. CPU-dependent tests then skip cleanly. EP coverage is
   preserved via the OGA suite. Skips lift automatically when ORT adds
   per-group QMoE support.

2. Each EP session pins a ~69 GB SHARED_CONSTANTS blob; two distinct ONNX
   graphs sequentially don't fit in 96 GB. `test_ort_dynamic_vs_fixed` is
   marked skip with the memory-budget reason.

Net effect: 3 pass + 7 skip / xfail.

Test coverage is provided by `BaseORTTests` (5 tests) + `BaseOGATests`
(4 tests) in conftest.py — see CLAUDE.md "Python Performance Tests".
"""

import pytest

from conftest import (
    BaseOGATests,
    BaseORTTests,
    ModelSpec,
    REPO_ROOT,
    normalize_drop_inputs_embeds,
    register_model_fixtures,
)

# ruff: noqa: F811


_CPU_SKIP_REASON = (
    "ORT CPU EP cannot serve gpt-oss-120b as a reference: either the "
    "QMoE op rejects per-group AWQ INT4 scales (group_size=32, "
    "hidden=2880 → 90 groups, last-dim mismatch [128,5760,1] vs "
    "[128,5760,90]) or repeated 64 GB CPU model loads have fragmented "
    "the heap into bad_alloc territory (Strix Halo has 96 GB total). "
    "EP coverage is preserved via the OGA suite."
)


def _cpu_skip_predicate(e):
    msg = str(e)
    qmoe = "QMoE" in msg and "fc1_experts_scales" in msg
    badalloc = "bad allocation" in msg or "bad_alloc" in msg
    return qmoe or badalloc


GPTOSS120B = ModelSpec(
    name="gptoss120b",
    model_dir=REPO_ROOT / "models" / "gpt-oss-120b-pergroup-asym-awq",
    onnx_file="model.onnx",
    data_files=["model.onnx.data"],
    hf_repo="amd/gpt-oss-120b-w-uint4-pergroup-asym-awq-onnx-fp16",
    num_layers=36,
    num_kv_heads=8,
    head_dim=64,
    has_position_ids=False,
    bos_token=199998,
    filler_tokens=[1234, 5678, 9012, 3456, 7890, 12345],
    oga_files=[
        "chat_template.jinja",
        "genai_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ],
    normalize_onnx_hook=normalize_drop_inputs_embeds,
    cpu_skip_predicate=_cpu_skip_predicate,
    cpu_skip_reason=_CPU_SKIP_REASON,
    # 120B SHARED_CONSTANTS blob is ~69 GB per session — module-scope
    # session reuse would exceed the 96 GB Strix Halo budget. Fall back to
    # per-test session creation to match the pre-2026-05 baseline.
    reuse_ep_session=False,
    reuse_oga_default_model=False,
    markers={
        "test_ort_dynamic_vs_fixed": pytest.mark.skip(
            reason=(
                "120B exceeds dynamic-vs-fixed memory budget. The test opens two "
                "EP sessions sequentially; each pins ~69 GB SHARED_CONSTANTS, so "
                "two distinct ONNX graphs need ~138 GB. Strix Halo has 96 GB. The "
                "dynamic-shape path is still exercised by the OGA suite and "
                "test_ort_dynamic_decode."
            )
        ),
        "test_ort_per_step_logits": pytest.mark.xfail(
            reason=(
                "gpt-oss MoE routing precision: decode KV cache has cosine ~0.94 "
                "after prefill, causing divergence by step 2. Same root cause as "
                "20B's test_ort_per_step_logits."
            ),
            strict=True,
        ),
        "test_oga_ep_chunked_prefill": pytest.mark.xfail(
            reason=(
                "gpt-oss MoE routing precision cascades across prefill chunks. "
                "Same root cause as 20B's test_oga_ep_chunked_prefill. Non-strict "
                "because the autotune-induced match-rate variance straddles the "
                "0.5 threshold."
            ),
            strict=False,
        ),
    },
)


(
    dynamic_model_path,
    fixed_decode_path,
    fixed_prefill_128_path,
    ep_dynamic_session,
    ep_fixed_decode_session,
    ep_fixed_prefill_128_session,
    oga_default_model,
) = register_model_fixtures(GPTOSS120B)


class TestGptOss120BORT(BaseORTTests):
    spec = GPTOSS120B


class TestGptOss120BOGA(BaseOGATests):
    spec = GPTOSS120B
