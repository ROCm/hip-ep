/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_EP_QSA_H
#define HIP_EP_QSA_H

#include "hip_custom_kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Experimental controlled attention boundary. No allocations or grid barriers.
 * Geometry: B=1, H=32, D=128, selector QH=4, groups of 4, budget 128 groups.
 * All pointers are contiguous device buffers on stream's device.
 * projected [B,S,640], q_weight/k_weight [128], features [B,S,4,128],
 * pool [B,P,128], tail [B,4,128] are BF16. cos/sin [65536,64] are FP32.
 * new K/V [B,S,4096] and caches [B,32,C,128] are FP16.
 * Only full prefill (past=0) followed by single-token decode is supported.
 */
HIP_KERNEL_API int hip_qsa_features(
    void *stream, const void *projected, const void *q_weight,
    const void *k_weight, const void *cos, const void *sin, void *features,
    void *pool, void *tail, const void *new_key, const void *new_value,
    void *keys, void *values, int batch, int sequence, int past, int capacity,
    int pool_capacity, float epsilon);

/* scores [B,chunk,P] FP32 and selected [B,chunk,128] int32 are caller-owned
 * persistent workspaces. Selection is causal, exact radix top-k; ties prefer
 * lower group IDs. first/chunk index queries within the current S-token call.
 */
HIP_KERNEL_API int hip_qsa_select(
    void *stream, const void *features, const void *pool, void *scores,
    void *selected, int batch, int sequence, int past, int first, int chunk,
    int pool_capacity);

/* Direct selected-address FP32 attention. No history gather/cast/copy.
 * query/output [B,S,4096], keys/values [B,32,C,128] are FP16.
 * partial [B,chunk,32,8,128], stats [B,chunk,32,8,2] are FP32 workspaces.
 */
HIP_KERNEL_API int hip_qsa_attend(
    void *stream, const void *query, const void *keys, const void *values,
    const void *selected, void *partial, void *stats, void *output,
    int batch, int sequence, int past, int first, int chunk, int capacity);

#ifdef __cplusplus
}
#endif
#endif
