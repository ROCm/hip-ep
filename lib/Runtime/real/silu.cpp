/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// SiLU / swish activation: y = x * sigmoid(x). Fused replacement for the
// exporter's separate onnx.Sigmoid + onnx.Mul (see SiluFusion.cpp).

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdio>

int wrap_silu(RuntimeState *state, void *input, void *output,
              int64_t num_elements, int64_t element_size_bytes) {
  OP_PROFILE(
      "silu",
      [&] {
        char b[48];
        snprintf(b, sizeof(b), "n=%lld:%s", (long long)num_elements,
                 (element_size_bytes == 2) ? "f16" : "f32");
        return std::string(b);
      },
      state);

  if (!state || !input || !output) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_silu: null required arg\n");
    return -1;
  }

  int hip_dtype;
  if (element_size_bytes == 2)
    hip_dtype = HIP_DTYPE_FLOAT16;
  else if (element_size_bytes == 4)
    hip_dtype = HIP_DTYPE_FLOAT32;
  else {
    fprintf(stderr, "[REAL] wrap_silu: unsupported element_size=%lld\n",
            (long long)element_size_bytes);
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  return hip_silu(stream, input, output, num_elements, hip_dtype);
}
