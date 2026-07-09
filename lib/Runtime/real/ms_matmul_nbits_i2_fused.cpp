/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Runtime wrapper for the fused DQ + MatMulNBits(bits=2) + Q kernel.
// Eliminates 2 global memory roundtrips vs running DQ, MatMul, Q separately.

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>

extern "C" int wrap_ms_matmul_nbits_i2_fused(
    RuntimeState *state,
    const void *A_u16,    // uint16 quantized input [M, K]
    const void *B,        // packed 2-bit weights [N, K/4]
    const void *w_scales, // float32 weight scales [N, k_blocks]
    const void *w_zp,     // uint8 weight zero points (nullable)
    const void *dq_scale, // float32 DQ scale scalar
    const void *dq_zp,    // uint16 DQ zero point scalar (nullable)
    const void *q_scale,  // float32 Q scale scalar
    const void *q_zp,     // uint16 Q zero point scalar (nullable)
    void *output,         // uint16 quantized output [M, N]
    int64_t M, int64_t N, int64_t K,
    int64_t block_size)
{
  OP_PROFILE(
      "ms_matmul_nbits_i2_fused",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "m=%lld,n=%lld,k=%lld,bs=%lld",
                 (long long)M, (long long)N, (long long)K,
                 (long long)block_size);
        return std::string(b);
      },
      state);

  if (!state || !A_u16 || !B || !w_scales || !dq_scale || !q_scale || !output) {
    fprintf(stderr, "[REAL] wrap_ms_matmul_nbits_i2_fused: null argument\n");
    return -1;
  }
  if (M <= 0 || N <= 0 || K <= 0) return 0;

  void *stream = hipdnn_ep_state_get_stream(state);
  if (!stream) {
    fprintf(stderr, "[REAL] wrap_ms_matmul_nbits_i2_fused: null stream\n");
    return -1;
  }

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_ms_matmul_nbits_i2_fused: M=%lld N=%lld K=%lld bs=%lld "
      "has_wzp=%d has_dqzp=%d has_qzp=%d\n",
      (long long)M, (long long)N, (long long)K, (long long)block_size,
      w_zp ? 1 : 0, dq_zp ? 1 : 0, q_zp ? 1 : 0);

  // Dispatch: M=1 (decode) → lean scalar kernel (no register spill)
  //           M>1 (prefill) → vectorized TILE_N template kernel
  return hip_matmul_nbits_i2_fused(
      stream, A_u16, B, w_scales, w_zp,
      dq_scale, dq_zp, q_scale, q_zp,
      output, M, N, K, block_size);
}
