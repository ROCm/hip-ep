/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Runtime wrapper of hip.token_major_gqa.

#include "../hipdnn_ep_runtime.h"
#include "hip_custom_kernels.h"
#include "kquant_abi.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>

int wrap_token_major_gqa(RuntimeState *state, const float *query,
                         const void *keyCache, const void *valueCache,
                         const void *mask, float *output, int64_t batchCount,
                         int64_t queryTokens, int64_t queryHeads,
                         int64_t kvHeads, int64_t cacheCapacity,
                         int64_t headDimension, int64_t maskBatchCount,
                         int64_t maskQueryTokens, int64_t outputQueryTokens,
                         float scale, const void *writtenRows,
                         int64_t writtenRowCount) {
  if (!state || !query || !keyCache || !valueCache || !mask || !output ||
      (writtenRows == nullptr) != (writtenRowCount == 0)) {
    std::fprintf(stderr, "wrap_token_major_gqa: null required argument\n");
    return -1;
  }
  if (batchCount <= 0 || queryTokens <= 0 || queryHeads <= 0 || kvHeads <= 0 ||
      cacheCapacity <= 0 ||
      !hip_token_major_gqa_head_dimension_valid(headDimension) ||
      queryHeads % kvHeads != 0 ||
      (maskBatchCount != 1 && maskBatchCount != batchCount) ||
      !std::isfinite(scale) || scale <= 0.0f) {
    std::fprintf(stderr, "wrap_token_major_gqa: invalid shape or scale\n");
    return -1;
  }
  if (queryTokens != maskQueryTokens || queryTokens != outputQueryTokens) {
    std::fprintf(stderr,
                 "wrap_token_major_gqa: query, mask, and output sequence "
                 "dimensions must match (query=%lld, mask=%lld, output=%lld)\n",
                 static_cast<long long>(queryTokens),
                 static_cast<long long>(maskQueryTokens),
                 static_cast<long long>(outputQueryTokens));
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  if (!stream) {
    std::fprintf(stderr, "wrap_token_major_gqa: null HIP stream\n");
    return -1;
  }

  // Scratch sized by the kernel library for this shape: the tile kernel's
  // parallel-block partials.
  const size_t scratchBytes =
      hip_token_major_gqa_scratch_size(batchCount, queryTokens, queryHeads,
                                       kvHeads, cacheCapacity, headDimension);
  void *scratch = nullptr;
  if (scratchBytes > 0) {
    if (hipdnn_ep_state_ensure_workspace(state, scratchBytes) != 0) {
      std::fprintf(stderr,
                   "wrap_token_major_gqa: cannot allocate %zu scratch "
                   "bytes\n",
                   scratchBytes);
      return -1;
    }
    scratch = hipdnn_ep_state_get_workspace(state);
    if (!scratch) {
      std::fprintf(stderr, "wrap_token_major_gqa: allocated "
                           "scratch is null\n");
      return -1;
    }
  }

  // The compiled graph bounds traversal through the destination rows of the
  // cache update when it has them; otherwise the whole compiled cache is
  // traversed and the mask rejects unwritten rows.
  return hip_token_major_gqa(
      stream, query, keyCache, valueCache, mask, output, batchCount,
      queryTokens, queryHeads, kvHeads, cacheCapacity,
      static_cast<const int64_t *>(writtenRows), writtenRowCount, headDimension,
      maskBatchCount, scale, scratch, scratchBytes);
}
