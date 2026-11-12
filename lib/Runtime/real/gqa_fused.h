/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_REAL_GQA_FUSED_H
#define HIPDNN_EP_REAL_GQA_FUSED_H

#include <hip/hip_runtime.h>

#include <cstdint>

struct RuntimeState;

// Fused causal fp16 attention of `sq` BSHD queries [B, sq, H, d] over a given
// BNSD cache [B, G, past_buf_seq, d] (the queries are its last sq rows) into
// a BSHD output [B, sq, H, d]. seqlens_k (int32 [B], total - 1) may be null:
// then all past_buf_seq rows are valid. Passing the same buffer as past_key
// and past_value lets decode read V from the K cache. Returns 0 on success,
// -1 on failure or a geometry/device the fused kernels do not cover.
int gqa_fused_attend_past(RuntimeState *state, hipStream_t stream,
                          const void *query, const void *past_key,
                          const void *past_value, const void *seqlens_k,
                          void *output, int64_t B, int64_t sq,
                          int64_t past_buf_seq, int64_t H, int64_t G, int64_t d,
                          float scale);

// int32 seqlens_k (kv_seqlen[b] - 1) from int64 kv_seqlen [B], written to the
// state-wide buffer so every layer passes the same pointer (the per-Compute
// seqlens_k cache is keyed on it). Returns that buffer, or null on failure.
const void *gqa_seqlens_from_lengths(RuntimeState *state, hipStream_t stream,
                                     const void *kv_seqlen, int64_t B);

// seqlens_k[0] (int32, total - 1) on the host for B == 1 through the
// per-Compute cache the fused attention reads too, so all layers sharing one
// seqlens_k buffer pay a single D2H. Returns -2 for B != 1 or on failure.
int32_t gqa_read_seqlens_k(RuntimeState *state, hipStream_t stream,
                           const void *seqlens_k, int64_t B);

#endif // HIPDNN_EP_REAL_GQA_FUSED_H
