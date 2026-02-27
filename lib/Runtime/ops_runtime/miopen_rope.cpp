/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- miopen_rope.cpp - hip.miopen.rope runtime
//---------------------------===//
//
// Rotary Positional Embeddings via MIOpen experimental API.
// Applies rotation in-place to Q and K tensors.
//
// Signature from MLIR lowering:
//   hip_miopen_rope(handle, q, k, cos_cache, sin_cache, start_pos)
//
// NOTE: MIOpen's RoPE API is experimental and may not be available on all
// ROCm versions.  This stub logs the call.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime_api.h>

extern "C" void hip_miopen_rope(void * /*handle*/, void *q, void *k,
                                void *cos_cache, void *sin_cache,
                                int64_t start_pos) {
  fprintf(stderr,
          "[hip_miopen_rope] called (q=%p, k=%p, cos=%p, sin=%p, pos=%lld)\n",
          q, k, cos_cache, sin_cache, (long long)start_pos);
}
