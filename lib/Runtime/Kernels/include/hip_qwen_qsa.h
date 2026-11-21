/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_EP_QWEN_QSA_H
#define HIP_EP_QWEN_QSA_H
#include "hip_qsa.h"
#ifdef __cplusplus
extern "C" {
#endif
HIP_KERNEL_API int hip_qwen_qsa_cast_hidden(void *stream, const void *hidden,
    const void *positions, void *converted, void *error_flag, int B, int S, int past);
HIP_KERNEL_API int hip_qwen_qsa_project_decode(void *stream, const void *hidden,
    const void *weight, const void *positions, void *projected, void *error_flag, int B, int past);
HIP_KERNEL_API int hip_qwen_qsa_features(void *stream, void *projected,
    const void *q_norm, const void *token_k_norm, const void *pool_k_norm,
    const void *cos, const void *sin, void *queries, void *pool, void *tail,
    int B, int S, int past, int capacity_groups);
HIP_KERNEL_API int hip_qwen_qsa_pack_kv(void *stream, const void *key,
    const void *value, void *cache_key, void *cache_value, int B, int S, int past, int C);
HIP_KERNEL_API int hip_qwen_qsa_select(void *stream, const void *queries,
    const void *pool, void *scores, void *selected, int B, int S, int past,
    int first, int count, int capacity_groups);
HIP_KERNEL_API int hip_qwen_qsa_attend(void *stream, const void *query,
    const void *key, const void *value, const void *selected, void *partial,
    void *stats, void *output, int B, int S, int past, int first, int count, int C);
#ifdef __cplusplus
}
#endif
#endif
