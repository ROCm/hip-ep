/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "error_check_macros.h"
#include "hip_custom_kernels.h"

#include <cstdio>

#define HIP_CHECK(cmd) HIP_CHECK_GOTO(cmd, cleanup)

int wrap_matmul_nbits(RuntimeState *state, const void *A, const void *B,
                      const void *scales, const void *zero_points,
                      const void *g_idx, const void *bias, void *output,
                      int64_t M, int64_t N, int64_t K, int64_t batch_count,
                      int64_t bits, int64_t block_size, int64_t elem_size) {
  if (!state || !A || !B || !scales || !output) {
    fprintf(stderr, "wrap_matmul_nbits: null argument\n");
    return -1;
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_matmul_nbits(M=%lld, N=%lld, K=%lld, "
                    "batch=%lld, bits=%lld, block_size=%lld, elem_size=%lld, "
                    "zero_points=%s, g_idx=%s, bias=%s)\n",
                    (long long)M, (long long)N, (long long)K,
                    (long long)batch_count, (long long)bits,
                    (long long)block_size, (long long)elem_size,
                    zero_points ? "yes" : "null", g_idx ? "yes" : "null",
                    bias ? "yes" : "null");

  void *stream = hipdnn_ep_state_get_stream(state);
  if (!stream) {
    fprintf(stderr, "wrap_matmul_nbits: null stream\n");
    return -1;
  }

  if (g_idx) {
    fprintf(stderr, "wrap_matmul_nbits: g_idx not supported\n");
    return -1;
  }

  int result = 0;
  HIP_CHECK(hip_matmul_nbits(stream, A, B, scales, zero_points, bias, output,
                             M, N, K, batch_count, bits, block_size,
                             elem_size));

cleanup:
  return result;
}
