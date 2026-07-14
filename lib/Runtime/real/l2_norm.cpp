/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// LpNormalization (ONNX-22) last-axis fast path, fused into ONE kernel.
//
//   p == 2: output = input / sqrt(sum(input^2) over the trailing axis)
//   p == 1: output = input / sum(|input|) over the trailing axis
//
// Replaces the Mul -> ReduceSum -> Sqrt -> Div decomposition (4-5
// dispatches) with a single block-per-row HIP kernel (see l2_norm_kernel.hip).
// Only the last (contiguous) axis is routed here; other axes stay on the
// decomposition path. FP32 accumulation; numerics match the decomposition.

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdio>

int wrap_l2_norm(RuntimeState *state, void *input, void *output,
                 int64_t input_num_elements, int64_t norm_size,
                 int64_t element_size_bytes, int64_t p) {
  OP_PROFILE(
      "l2norm",
      [&] {
        char b[64];
        int64_t outer =
            norm_size > 0 ? input_num_elements / norm_size : 0;
        snprintf(b, sizeof(b), "%lldx%lld:%s:p%lld", (long long)outer,
                 (long long)norm_size,
                 (element_size_bytes == 2) ? "f16" : "f32", (long long)p);
        return std::string(b);
      },
      state);

  if (!state || !input || !output) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_l2_norm: null required arg\n");
    return -1;
  }
  if (norm_size <= 0) {
    fprintf(stderr, "[REAL] wrap_l2_norm: norm_size=%lld\n",
            (long long)norm_size);
    return -1;
  }
  if (input_num_elements % norm_size != 0) {
    fprintf(stderr,
            "[REAL] wrap_l2_norm: input_num(%lld) not divisible by "
            "norm_size(%lld)\n",
            (long long)input_num_elements, (long long)norm_size);
    return -1;
  }
  if (p != 1 && p != 2) {
    fprintf(stderr, "[REAL] wrap_l2_norm: unsupported p=%lld\n", (long long)p);
    return -1;
  }

  int hip_dtype;
  if (element_size_bytes == 2) {
    hip_dtype = HIP_DTYPE_FLOAT16;
  } else if (element_size_bytes == 4) {
    hip_dtype = HIP_DTYPE_FLOAT32;
  } else {
    fprintf(stderr,
            "[REAL] wrap_l2_norm: unsupported element_size=%lld "
            "(supported: 2=fp16, 4=fp32)\n",
            (long long)element_size_bytes);
    return -1;
  }

  int64_t outer = input_num_elements / norm_size;

  void *stream = hipdnn_ep_state_get_stream(state);
  RUNTIME_DEBUG_LOG("[REAL] wrap_l2_norm: outer=%lld, norm_size=%lld, "
                    "elem_size=%lld, p=%lld\n",
                    (long long)outer, (long long)norm_size,
                    (long long)element_size_bytes, (long long)p);

  return hip_l2_norm(stream, input, output, outer, norm_size,
                     static_cast<int>(p), hip_dtype);
}
