/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include <string>
#include "gqa_paged_attention.h"
namespace hipdnn_ep {
struct PagedAttentionResult { GqaPagedConfig config; std::string source; };
// Host-side preparation, outside timed launches. The caller's exact native
// case key includes every request's q/past, page layout and FP16/RoPE options.
HIP_KERNEL_API PagedAttentionResult resolvePagedAttention(const hip_paged_attention_f16_desc&,const std::string& shape);
// Normal production resolver/preparation. A compatible lookup is immediate;
// a miss or online_only explicitly prepares/tunes before configured calls.
HIP_KERNEL_API int preparePagedAttention(void* stream,const hip_paged_attention_f16_desc&,
    void* workspace,size_t workspace_bytes,const std::string& shape,
    bool online_only,PagedAttentionResult* result);
}
