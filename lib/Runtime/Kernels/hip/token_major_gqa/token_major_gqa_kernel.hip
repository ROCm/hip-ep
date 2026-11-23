/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// hip_token_major_gqa: its arguments checked, the call run on
// llama.cpp's tile kernel (token_major_gqa_tile.hip).

#include "hip_custom_kernels.h"

#include "token_major_gqa_tile.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

token_major_gqa::TileAttentionArgs
tileArguments(int64_t batchCount, int64_t queryTokens, int64_t queryHeads,
              int64_t kvHeads, int64_t cacheCapacity, int64_t headDimension,
              int64_t maskBatchCount) {
  token_major_gqa::TileAttentionArgs args{};
  args.batchCount = batchCount;
  args.queryTokens = queryTokens;
  args.queryHeads = queryHeads;
  args.kvHeads = kvHeads;
  args.cacheCapacity = cacheCapacity;
  args.liveKeys = cacheCapacity;
  args.headDimension = headDimension;
  args.maskBatchCount = maskBatchCount;
  return args;
}

} // namespace

extern "C" size_t hip_token_major_gqa_scratch_size(
    int64_t batchCount, int64_t queryTokens, int64_t queryHeads,
    int64_t kvHeads, int64_t cacheCapacity, int64_t headDimension) {
  if (batchCount <= 0 || queryTokens <= 0 || queryHeads <= 0 || kvHeads <= 0 ||
      cacheCapacity <= 0 || queryHeads % kvHeads != 0 ||
      !hip_token_major_gqa_head_dimension_valid(headDimension))
    return 0;
  token_major_gqa::TileAttentionArgs args =
      tileArguments(batchCount, queryTokens, queryHeads, kvHeads,
                    cacheCapacity, headDimension, /*maskBatchCount=*/1);
  // Only the shape matters for the plan; a non-null pointer stands for the
  // mask.
  args.mask = reinterpret_cast<const __half *>(&args);
  return token_major_gqa::tileAttentionScratchBytes(args);
}

extern "C" int hip_token_major_gqa(
    void *stream, const float *query, const void *keyCache,
    const void *valueCache, const void *mask, float *output,
    int64_t batchCount, int64_t queryTokens, int64_t queryHeads,
    int64_t kvHeads, int64_t cacheCapacity, const int64_t *writtenRows,
    int64_t writtenRowCount, int64_t headDimension, int64_t maskBatchCount,
    float scale, void *scratch, size_t scratchSize) {
  if (!query || !keyCache || !valueCache || !mask || !output) {
    std::fprintf(stderr,
                 "[custom_kernels] hip_token_major_gqa: null tensor\n");
    return -1;
  }
  if (batchCount <= 0 || queryTokens <= 0 || queryHeads <= 0 || kvHeads <= 0 ||
      cacheCapacity <= 0 || writtenRowCount < 0 ||
      (writtenRows == nullptr) != (writtenRowCount == 0) ||
      !hip_token_major_gqa_head_dimension_valid(headDimension) ||
      queryHeads % kvHeads != 0 ||
      (maskBatchCount != 1 && maskBatchCount != batchCount) ||
      !std::isfinite(scale) || scale <= 0.0f) {
    std::fprintf(stderr,
                 "[custom_kernels] hip_token_major_gqa: invalid shape or "
                 "scale\n");
    return -1;
  }
  token_major_gqa::TileAttentionArgs args =
      tileArguments(batchCount, queryTokens, queryHeads, kvHeads,
                    cacheCapacity, headDimension, maskBatchCount);
  args.stream = static_cast<hipStream_t>(stream);
  args.query = query;
  args.keyCache = static_cast<const __half *>(keyCache);
  args.valueCache = static_cast<const __half *>(valueCache);
  args.mask = static_cast<const __half *>(mask);
  args.output = output;
  args.writtenRows = writtenRows;
  args.writtenRowCount = writtenRowCount;
  args.scale = scale;
  args.scratch = scratch;
  args.scratchBytes = scratchSize;
  const int status = token_major_gqa::launchTileAttention(args);
  if (status == token_major_gqa::kTileUnsupported) {
    std::fprintf(stderr, "[custom_kernels] hip_token_major_gqa: no tile "
                         "kernel takes this call\n");
    return -1;
  }
  return status;
}
