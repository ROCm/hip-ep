/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- gather.cpp - Embedding table lookup (stub) -------------------------===//
//
// Stub implementation: zeroes the output buffer.
// TODO: implement as a HIP kernel that copies table[indices[i]] -> output[i].
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime_api.h>

extern "C" void hip_gather(void* /*handle*/, void* /*indices*/,
                           void* /*table*/, void* output) {
  // Stub: cannot determine output size without descriptor metadata.
  // In a real implementation the sizes would be passed or inferred.
  (void)output;
  fprintf(stderr,
          "[hip_gather] stub called -- output not zeroed (no size info)\n");
}
