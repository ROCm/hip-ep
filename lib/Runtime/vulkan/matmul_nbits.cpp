/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../hipdnn_ep_runtime.h"
#include "vulkan_kernels.h"

#include <cstdio>

int wrap_matmul_nbits(RuntimeState *state, const void *A, const void *B,
                      const void *scales, const void *zero_points,
                      const void *g_idx, const void *bias, void *output,
                      int64_t M, int64_t N, int64_t K, int64_t batch_count,
                      int64_t bits, int64_t block_size, int64_t elem_size) {

  if (!A || !B || !scales || !output) {
    fprintf(stderr, "wrap_matmul_nbits[vulkan]: null argument\n");
    return -1;
  }

  if (g_idx) {
    fprintf(stderr, "wrap_matmul_nbits[vulkan]: g_idx not supported\n");
    return -1;
  }

  // In the EP flow, all pointers are GPU-resident (from hipMalloc/pool).
  // Use the zero-copy GPU-pointer mode for optimal performance.
  return vulkan_matmul_nbits_gpu(A, B, scales, zero_points, bias, output, M, N,
                                 K, batch_count, bits, block_size, elem_size);
}
