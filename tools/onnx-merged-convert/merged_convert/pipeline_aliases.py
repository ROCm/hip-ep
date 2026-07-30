#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

from __future__ import annotations

from .int8kv import merge_pipeline
from .qdq_ext import (
    CONVERT_PROFILE_LOW_BIT_INTERNAL,
    convert_lm_head_model_q,
    merge_quantized_pipeline_models,
    patch_emb_model_q,
    process_one_onnx_q,
)
from .step5_merge import merge_split_pipeline_models

CONVERT_PROFILE_LOW_BIT = CONVERT_PROFILE_LOW_BIT_INTERNAL
merge_split_pipeline = merge_split_pipeline_models
merge_split_pipeline_quantized = merge_pipeline
merge_split_pipeline_lowbit = merge_quantized_pipeline_models
patch_emb_quantized = patch_emb_model_q
convert_head_quantized = convert_lm_head_model_q
process_one_onnx = process_one_onnx_q
