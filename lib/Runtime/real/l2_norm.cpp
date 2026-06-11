/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// L2 normalization over the last axis:
//   y = x * rsqrt(sum(x^2, last_axis) + epsilon)
// Fused replacement for the exporter q/k normalization chain (see
// l2_norm_kernel.hip and L2NormFusion.cpp).

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdio>

int wrap_l2_normalize(RuntimeState *state, void *input, void *output,
                      int64_t input_num_elements, int64_t norm_size,
                      int64_t element_size_bytes, float epsilon) {
  OP_PROFILE(
      "l2norm",
      [&] {
        char b[64];
        int64_t outer = norm_size > 0 ? input_num_elements / norm_size : 0;
        snprintf(b, sizeof(b), "%lldx%lld:%s", (long long)outer,
                 (long long)norm_size,
                 (element_size_bytes == 2) ? "f16" : "f32");
        return std::string(b);
      },
      state);

  if (!state || !input || !output) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_l2_normalize: null required arg\n");
    return -1;
  }
  if (norm_size <= 0 || input_num_elements % norm_size != 0) {
    fprintf(stderr,
            "[REAL] wrap_l2_normalize: bad norm_size=%lld input_num=%lld\n",
            (long long)norm_size, (long long)input_num_elements);
    return -1;
  }

  int hip_dtype;
  if (element_size_bytes == 2)
    hip_dtype = HIP_DTYPE_FLOAT16;
  else if (element_size_bytes == 4)
    hip_dtype = HIP_DTYPE_FLOAT32;
  else {
    fprintf(stderr, "[REAL] wrap_l2_normalize: unsupported element_size=%lld\n",
            (long long)element_size_bytes);
    return -1;
  }

  int64_t outer = input_num_elements / norm_size;
  void *stream = hipdnn_ep_state_get_stream(state);
  return hip_l2_normalize(stream, input, output, outer, norm_size, epsilon,
                          hip_dtype);
}
