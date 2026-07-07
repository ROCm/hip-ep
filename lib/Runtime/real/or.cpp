/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Or: y = a || b (element-wise logical OR on bool tensors).
//
// Source:
// onnxruntime/core/providers/cuda/math/binary_elementwise_ops_impl.{h,cu}
//         @ v1.22.2 (BINARY_OP_NAME_EXPR(Or, (a | b)),
//                    SPECIALIZED_BINARY_ELEMENTWISE_IMPL(Or, bool))
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>

int wrap_or(RuntimeState *state, void *a, void *b, void *output,
            int64_t num_elements, int64_t data_type) {
  OP_PROFILE(
      "or",
      [&] {
        char b[48];
        snprintf(b, sizeof(b), "%lld:%s", (long long)num_elements,
                 hipdnn_ep_datatype_name(data_type));
        return std::string(b);
      },
      state);

  if (!state || !a || !b || !output) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_or: null argument\n");
    return -1;
  }
  if (num_elements <= 0) {
    return 0;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  RUNTIME_DEBUG_LOG("[REAL] wrap_or: num=%lld, data_type=%s "
                    "(treated as 1-byte bool) -> hip_elementwise_or\n",
                    (long long)num_elements,
                    hipdnn_ep_datatype_name(data_type));
  return hip_elementwise_or(stream, a, b, output, num_elements);
}
