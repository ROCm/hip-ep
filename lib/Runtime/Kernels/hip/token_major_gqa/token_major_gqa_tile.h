/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// llama.cpp's tiled flash attention (ggml-cuda/fattn-tile.cuh) behind the
// hip_token_major_gqa contract. Internal to the kernel library:
// hip_token_major_gqa validates its arguments, then runs each call here.

#ifndef HIP_TOKEN_MAJOR_GQA_TILE_H
#define HIP_TOKEN_MAJOR_GQA_TILE_H

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace token_major_gqa {

struct TileAttentionArgs {
  hipStream_t stream;
  const float *query;     // [B][T][H][D]
  const __half *keyCache; // [B][capacity][G][D]
  const __half *valueCache;
  const __half *mask; // [maskBatchCount][T][capacity] additive, or null
  float *output;      // [B][T][H][D]
  int64_t batchCount;
  int64_t queryTokens;
  int64_t queryHeads;
  int64_t kvHeads;
  int64_t cacheCapacity;
  int64_t liveKeys;
  const int64_t *writtenRows;
  int64_t writtenRowCount;
  int64_t headDimension;
  int64_t maskBatchCount;
  float scale;
  void *scratch;
  size_t scratchBytes;
};

// Runs the call on llama.cpp's tile kernel. Returns 0 on success, a HIP
// error code on a launch failure, and kTileUnsupported when this path has no
// instantiation for the shape.
constexpr int kTileUnsupported = -2;
int launchTileAttention(const TileAttentionArgs &args);

// The scratch launchTileAttention wants for the call (it runs with less, down
// to two parallel blocks), or 0 when it does not take the call.
size_t tileAttentionScratchBytes(const TileAttentionArgs &args);

} // namespace token_major_gqa

#endif // HIP_TOKEN_MAJOR_GQA_TILE_H
