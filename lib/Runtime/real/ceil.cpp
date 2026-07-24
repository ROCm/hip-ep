/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Ceil: y = ceil(x) (element-wise ceiling).
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>

static int ceil_hipdnn_to_hip_dtype(int64_t hipdnn_type) {
  switch (hipdnn_type) {
  case HIPDNN_EP_DATATYPE_HALF:
    return HIP_DTYPE_FLOAT16;
  case HIPDNN_EP_DATATYPE_FLOAT:
    return HIP_DTYPE_FLOAT32;
  default:
    return -1;
  }
}

int wrap_ceil(RuntimeState *state, void *input, void *output,
              int64_t num_elements, int64_t data_type) {
  OP_PROFILE(
      "ceil",
      [&] {
        char b[48];
        snprintf(b, sizeof(b), "%lld:%s", (long long)num_elements,
                 hipdnn_ep_datatype_name(data_type));
        return std::string(b);
      },
      state);

  if (!state || !input || !output) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_ceil: null argument\n");
    return -1;
  }
  if (num_elements <= 0) {
    return 0;
  }

  int hip_dtype = ceil_hipdnn_to_hip_dtype(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr,
            "[REAL] wrap_ceil: unsupported data_type=%s(%lld) "
            "(supported: f16, f32)\n",
            hipdnn_ep_datatype_name(data_type), (long long)data_type);
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_ceil: num=%lld, data_type=%s -> hip_elementwise_ceil\n",
      (long long)num_elements, hipdnn_ep_datatype_name(data_type));
  return hip_elementwise_ceil(stream, input, output, num_elements, hip_dtype);
}
