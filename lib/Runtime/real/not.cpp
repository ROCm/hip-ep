/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Not: y = !x (bool/INT8, 1 byte per element).
//
// Source: onnxruntime/core/providers/cuda/math/unary_elementwise_ops_impl.cu
//         @ v1.22.2 (UNARY_OP_NAME_EXPR(Not, !a),
//                    SPECIALIZED_UNARY_ELEMENTWISE_IMPL(Not, bool))
//
// The data_type argument is logged but otherwise ignored -- the kernel
// always treats input/output as 1-byte streams (matching ORT's bool spec).
#include "../cpu_fallback_invoke.h"
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>

int wrap_not(RuntimeState *state, void *input, void *output,
             int64_t num_elements, int64_t data_type) {
  OP_PROFILE(
      "not",
      [&] {
        char b[48];
        snprintf(b, sizeof(b), "%lld:%s", (long long)num_elements,
                 hipdnn_ep_datatype_name(data_type));
        return std::string(b);
      },
      state);

  if (!state || !input || !output) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_not: null argument\n");
    return -1;
  }
  if (num_elements <= 0) {
    return 0;
  }

  // ONNX Not is bool-only. The hip-to-llvm lowering for i1 types passes
  // data_type = 0 (no enum slot for i1) -- accept any value, since the
  // wire format is always a 1-byte stream when the graph type was tensor<i1>.
  // Logged for debugging but not enforced.
  (void)data_type;

  void *stream = hipdnn_ep_state_get_stream(state);
  {
    const int64_t bool_dtype = HIPDNN_EP_DATATYPE_UINT8;
    const int fb_rc =
        hipdnn_cpu_fb_try_unary_1d(state, stream, "Not", input, output,
                                   num_elements, bool_dtype);
    if (fb_rc == 0)
      return 0;
    if (fb_rc < 0)
      return -1;
  }
  RUNTIME_DEBUG_LOG("[REAL] wrap_not: num=%lld, data_type=%s "
                    "(treated as 1-byte bool) -> hip_elementwise_not\n",
                    (long long)num_elements,
                    hipdnn_ep_datatype_name(data_type));
  return hip_elementwise_not(stream, input, output, num_elements);
}
