/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>

#include <cstdio>

int wrap_dequant_bnb4(RuntimeState *state, void *packed, void *absmax,
                      void *output, int64_t n, int64_t k, int64_t block_size,
                      int64_t quant_type) {
  OP_PROFILE(
      "dequant_bnb4",
      [&] {
        char b[96];
        snprintf(b, sizeof(b), "N=%lld K=%lld block=%lld q=%lld", (long long)n,
                 (long long)k, (long long)block_size, (long long)quant_type);
        return std::string(b);
      },
      state);
  if (!state || !packed || !absmax || !output || n <= 0 || k <= 0 ||
      block_size <= 0) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_dequant_bnb4: bad argument\n");
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  return hip_dequant_bnb4(stream, packed, absmax, output, n, k, block_size,
                          quant_type);
}
